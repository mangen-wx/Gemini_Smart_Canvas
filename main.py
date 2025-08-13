import base64
import os
import uuid
import re
from typing import Optional, AsyncGenerator

import aiohttp
import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register


@register("Gemini_Smart_Canvas", "沐沐沐倾丶", "Gemini智能绘图", "1.0.0") # 移除 description 参数
class GeminiImageGenerator(Star):
    """
    Gemini 图片生成与编辑插件。
    提供文生图和图生图功能，支持提示词扩展和自定义反向提示词。
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        """
        插件初始化，加载配置并设置必要的资源。
        """
        super().__init__(context)
        self.config = config

        logger.info("Gemini智能绘图插件初始化成功，开始加载配置...")

        self.api_keys = self.config.get("gemini_api_keys", [])
        if not self.api_keys:
            logger.error("错误：未检测到Gemini API密钥配置，请检查插件设置。")

        # 初始化图片临时存储目录
        plugin_dir = os.path.dirname(__file__)
        self.save_dir = os.path.join(plugin_dir, "temp_images")
        os.makedirs(self.save_dir, exist_ok=True) # 使用 exist_ok=True 避免重复创建错误
        logger.info(f"图片临时存储目录已准备就绪: {self.save_dir}")

        # 加载API基础URL，并确保格式正确
        self.api_base_url = self._normalize_api_base_url(
            self.config.get("api_base_url", "https://generativelanguage.googleapis.com")
        )

        # 加载模型名称配置
        self.text_model_name = self.config.get("text_model_name", "gemini-2.5-flash")
        self.image_model_name = self.config.get("image_model_name", "gemini-2.0-flash-preview-image-generation")

        # 新增：固定正向提示词前缀
        self.fixed_positive_prefix = self.config.get("fixed_positive_prefix", "")

        # 新增：是否启用提示词增强功能
        self.enable_prompt_enhancement = self.config.get("enable_prompt_enhancement", True)

        # 从配置中加载反向提示词和提示词扩展模板
        self.default_negative_prompt = self.config.get("default_negative_prompt", "")
        self.prompt_expansion_template = self.config.get("prompt_expansion_template", "")

    def _normalize_api_base_url(self, url: str) -> str:
        """规范化API基础URL，确保以https://开头且不以/结尾。"""
        url = url.strip()
        if not url.startswith("https://"):
            url = f"https://{url}"
        if url.endswith("/"):
            url = url[:-1]
        return url

    def _get_current_api_key(self) -> Optional[str]:
        """
        获取当前使用的 API 密钥。
        始终返回列表中的第一个密钥。
        """
        if not self.api_keys:
            return None
        return self.api_keys[0]

    async def _send_api_request(self, endpoint: str, payload: dict) -> dict:
        """
        通用异步发送API请求的方法。
        """
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url=endpoint, json=payload, headers=headers) as response:
                    response.raise_for_status()  # 抛出HTTPError如果状态码不是2xx
                    data = await response.json()
                    logger.debug(f"Gemini API响应数据: {data}")
                    return data
            except aiohttp.ClientResponseError as e:
                logger.error(f"Gemini API请求失败: HTTP {e.status}, 响应: {e.message}")
                raise Exception(f"API请求失败: HTTP {e.status} - {e.message}")
            except aiohttp.ClientError as e:
                logger.error(f"Gemini API网络请求失败: {e}")
                raise Exception(f"API网络请求失败: {e}")
            except Exception as e:
                logger.error(f"异步API请求过程中发生未知错误: {e}")
                raise Exception(f"API请求发生未知错误: {e}")

    async def _parse_image_response(self, data: dict, temp_prefix: str) -> Optional[bytes]:
        """
        解析Gemini图片API响应，提取图片数据或从URL下载图片。
        """
        image_data = None
        image_url = None

        if "candidates" in data and len(data["candidates"]) > 0:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    base64_str = part["inlineData"]["data"].replace("\n", "").replace("\r", "")
                    image_data = base64.b64decode(base64_str)
                    logger.debug("成功从inlineData中提取图片数据。")
                    break
                elif "text" in part:
                    extracted_url = self._extract_image_url_from_text(part["text"])
                    if extracted_url:
                        image_url = extracted_url
                        logger.debug(f"在文本部分发现图片URL: {image_url}")
                        break
            
            if not image_data and not image_url:
                logger.warning(f"Gemini图片API响应中缺少'inlineData'或可识别的图片URL: {data}")
        else:
            logger.warning(f"Gemini图片API响应中缺少'candidates'或内容部分: {data}")

        if image_data:
            return image_data
        elif image_url:
            temp_download_path = os.path.join(self.save_dir, f"{temp_prefix}_{uuid.uuid4()}.png")
            try:
                await self._download_image_from_url(image_url, temp_download_path)
                with open(temp_download_path, "rb") as f:
                    downloaded_image_bytes = f.read()
                logger.debug(f"成功从URL下载图片: {image_url}")
                return downloaded_image_bytes
            except Exception as download_e:
                logger.error(f"从URL下载图片失败 {image_url}: {download_e}")
                raise Exception(f"图片操作成功，但从URL下载图片失败: {download_e}")
            finally:
                if os.path.exists(temp_download_path):
                    os.remove(temp_download_path)
                    logger.debug(f"已清理临时下载文件: {temp_download_path}")
        else:
            return None # 如果没有图片数据也没有URL，返回None

    async def _call_gemini_image_api(self, prompt: str, api_key: str, image_base64: Optional[str] = None) -> Optional[bytes]:
        """
        调用Gemini图片生成/编辑API的通用方法。
        """
        model_name = self.image_model_name
        endpoint = f"{self.api_base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        logger.debug(f"正在向Gemini图片模型发送异步请求: {endpoint}")

        parts = [{"text": prompt}]
        if image_base64:
            parts.append({"inlineData": {"mimeType": "image/png", "data": image_base64}})
        
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 2048,
            },
        }

        data = await self._send_api_request(endpoint, payload)
        return await self._parse_image_response(data, "downloaded_image")

    async def _generate_image_with_retry(self, prompt: str) -> Optional[bytes]:
        """
        带重试逻辑的图片生成方法。
        """
        current_key = self._get_current_api_key()
        if not current_key:
            raise Exception("未配置有效的Gemini API密钥。")

        logger.info(f"尝试生成图片（使用密钥：{current_key[:5]}...）")
        return await self._call_gemini_image_api(prompt, current_key)

    async def _edit_image_with_retry(self, prompt: str, image_path: str) -> Optional[bytes]:
        """
        带重试逻辑的图片编辑方法。
        """
        current_key = self._get_current_api_key()
        if not current_key:
            raise Exception("未配置有效的Gemini API密钥。")

        logger.info(f"尝试编辑图片（使用密钥：{current_key[:5]}...）")
        
        # 读取图片并转换为Base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8").replace("\n", "").replace("\r", "")

        return await self._call_gemini_image_api(prompt, current_key, image_base64)

    @filter.command("gemini_image", alias={"文生图"})
    async def generate_image(self, event: AstrMessageEvent, prompt: str) -> AsyncGenerator[Comp.MessageComponent, None]:
        """
        根据文本描述生成图片。
        该方法会根据配置决定是否扩展用户提示词，并结合反向提示词进行图片生成。
        """
        if not self.api_keys:
            yield event.plain_result("错误：插件未配置有效的Gemini API密钥。")
            return

        if not prompt.strip():
            yield event.plain_result("请提供您想要生成的图片描述，例如：/文生图 一只戴帽子的猫在月球上。")
            return

        save_path = None
        try:
            final_positive_prompt = prompt
            prompt_to_send_to_model = ""

            if self.enable_prompt_enhancement:
                # 尝试扩展提示词
                expanded_positive_prompt = await self._expand_prompt_with_gemini(prompt)
                if expanded_positive_prompt:
                    final_positive_prompt = expanded_positive_prompt
                    logger.info(f"原始提示词: '{prompt}'")
                    logger.info(f"扩展后正向提示词: '{expanded_positive_prompt}'")
                else:
                    logger.warning("提示词扩展服务异常，将使用原始描述进行图片生成。")
                
                # 强制添加固定正向提示词前缀
                if self.fixed_positive_prefix:
                    final_positive_prompt = f"{self.fixed_positive_prefix}, {final_positive_prompt}"
                    logger.info(f"已添加固定正向提示词前缀: '{self.fixed_positive_prefix}'")

                # 结合反向提示词
                prompt_to_send_to_model = final_positive_prompt
                if self.default_negative_prompt:
                    prompt_to_send_to_model += f", avoid {self.default_negative_prompt}"
            else:
                logger.info("提示词增强功能已禁用，将直接使用用户原始提示词，不添加任何额外提示词。")
                prompt_to_send_to_model = prompt # 禁用增强时，只用原始提示词
            
            logger.info(f"发送给图片模型的最终提示词: '{prompt_to_send_to_model}'")
            yield event.plain_result("正在生成图片，请稍候...")

            image_data = await self._generate_image_with_retry(prompt_to_send_to_model)

            if not image_data:
                logger.error("图片生成失败：API密钥可能无效或已达到使用限制。")
                yield event.plain_result("图片生成失败：请检查API密钥或稍后再试。")
                return

            # 保存图片
            file_name = f"{uuid.uuid4()}.png"
            save_path = os.path.join(self.save_dir, file_name)
            with open(save_path, "wb") as f:
                f.write(image_data)

            logger.info(f"生成的图片已临时保存至: {save_path}")
            yield event.chain_result([Image.fromFileSystem(save_path)])
            logger.info(f"图片发送成功。")

        except Exception as e:
            logger.error(f"图片生成过程中发生异常: {e}", exc_info=True)
            yield event.plain_result(f"图片生成失败，错误信息: {e}")

        finally:
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已清理临时图片文件: {save_path}")
                except Exception as e:
                    logger.warning(f"清理临时图片文件失败: {e}")

    @filter.command("gemini_edit", alias={"图编辑"})
    async def edit_image(self, event: AstrMessageEvent, prompt: str) -> AsyncGenerator[Comp.MessageComponent, None]:
        """
        仅支持：引用图片后发送指令编辑图片。
        编辑时只使用用户提供的提示词，不进行任何增强。
        """
        if not self.api_keys:
            yield event.plain_result("错误：插件未配置有效的Gemini API密钥。")
            return

        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result("未检测到引用的图片。请长按图片选择“回复”后，再发送编辑指令。")
            return

        # 图片编辑不进行任何增强，只使用用户原始提示词
        final_prompt_for_model = prompt
        logger.info("图片编辑功能不进行提示词增强，将直接使用用户原始提示词。")

        logger.info(f"发送给图片编辑模型的最终提示词: '{final_prompt_for_model}'")
        yield event.plain_result(f"正在编辑图片，请稍候...")
        
        async for result in self._process_image_edit(event, final_prompt_for_model, image_path):
            yield result

    @filter.llm_tool(name="edit_image",
                     description="编辑现有图片。当你需要编辑图片时，请使用此工具。\n"
                                 "你可以通过描述图片内容的修改来触发此工具，例如：\n"
                                 "- \"把猫咪改成黑色\"\n"
                                 "- \"肤色改为白色\"\n"
                                 "- \"背景换一下\"\n"
                                 "- \"添加一朵花\"\n"
                                 "- \"让人物穿上红色的衣服\"\n"
                                 "- \"把天空变成蓝色\"\n"
                                 "- \"移除图片中的文字\"")
    async def edit_image_tool(self, event: AstrMessageEvent, prompt: str) -> AsyncGenerator[Comp.MessageComponent, None]:
        """
        LLM工具接口：编辑现有图片。
        Args:
            prompt(string): 编辑描述（例如：把猫咪改成黑色）
        """
        if not self.api_keys:
            yield event.plain_result("错误：插件未配置有效的Gemini API密钥。")
            return

        if not prompt.strip():
            yield event.plain_result("请提供图片编辑的具体描述，例如：把猫咪改成黑色。")
            return

        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result("未检测到引用的图片。请长按图片选择“回复”后，再发送编辑指令。")
            return

        # 图片编辑不进行任何增强，只使用用户原始提示词
        final_prompt_for_model = prompt
        logger.info("图片编辑功能不进行提示词增强，将直接使用用户原始提示词。")

        logger.info(f"发送给图片编辑模型的最终提示词: '{final_prompt_for_model}'")
        yield event.plain_result(f"正在编辑图片，请稍候...")
        
        async for result in self._process_image_edit(event, final_prompt_for_model, image_path):
            yield result

    @filter.llm_tool(name="generate_image",
                     description="根据文本描述生成图片，当你需要生成图片时请使用此工具。")
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str) -> AsyncGenerator[Comp.MessageComponent, None]:
        """
        LLM工具接口：根据文本描述生成图片。
        Args:
            prompt(string): 图片描述文本（例如：画只猫）
        """
        async for result in self.generate_image(event, prompt):
            yield result

    async def _extract_image_from_reply(self, event: AstrMessageEvent) -> Optional[str]:
        """
        从回复消息中提取图片并返回本地路径。
        """
        try:
            message_components = event.message_obj.message
            for comp in message_components:
                if isinstance(comp, Comp.Reply):
                    for quoted_comp in comp.chain:
                        if isinstance(quoted_comp, Comp.Image):
                            image_path = await quoted_comp.convert_to_file_path()
           
