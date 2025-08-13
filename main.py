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


@register("Gemini_Smart_Canvas", "沐沐沐倾丶", "Gemini智能绘图", "1.0.0",
          description="基于Gemini模型，提供智能文生图和图生图功能，支持提示词自动扩展和固定反向提示词，旨在为用户提供高质量、富有创意的图像生成体验。")
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

        # 新增：是否启用提示词增强功能
        self.enable_prompt_enhancement = self.config.get("enable_prompt_enhancement", True)

        # 从配置中加载反向提示词和提示词扩展模板，不再硬编码默认值
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
            final_prompt_for_model = prompt
            if self.enable_prompt_enhancement:
                # 尝试扩展提示词
                expanded_positive_prompt = await self._expand_prompt_with_gemini(prompt)
                if expanded_positive_prompt:
                    final_prompt_for_model = expanded_positive_prompt
                    logger.info(f"原始提示词: '{prompt}'")
                    logger.info(f"扩展后正向提示词: '{expanded_positive_prompt}'")
                else:
                    logger.warning("提示词扩展服务异常，将使用原始描述进行图片生成。")
                
                # 结合反向提示词
                if self.default_negative_prompt:
                    final_prompt_for_model += f", avoid {self.default_negative_prompt}"
            else:
                logger.info("提示词增强功能已禁用，将直接使用用户原始提示词。")
            
            logger.info(f"发送给图片模型的最终提示词: '{final_prompt_for_model}'")
            yield event.plain_result("正在生成图片，请稍候...")

            image_data = await self._generate_image_with_retry(final_prompt_for_model)

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
        编辑时根据配置决定是否附加反向提示词。
        """
        if not self.api_keys:
            yield event.plain_result("错误：插件未配置有效的Gemini API密钥。")
            return

        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result("未检测到引用的图片。请长按图片选择“回复”后，再发送编辑指令。")
            return

        final_prompt_for_model = prompt
        if self.enable_prompt_enhancement and self.default_negative_prompt:
            final_prompt_for_model += f", avoid {self.default_negative_prompt}"
            logger.info("提示词增强功能已启用，编辑时将添加反向提示词。")
        else:
            logger.info("提示词增强功能已禁用或未配置反向提示词，编辑时将直接使用用户原始提示词。")

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

        final_prompt_for_model = prompt
        if self.enable_prompt_enhancement and self.default_negative_prompt:
            final_prompt_for_model += f", avoid {self.default_negative_prompt}"
            logger.info("提示词增强功能已启用，编辑时将添加反向提示词。")
        else:
            logger.info("提示词增强功能已禁用或未配置反向提示词，编辑时将直接使用用户原始提示词。")

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
                            logger.debug(f"从回复中提取到图片并处理为本地路径：{image_path}")
                            return image_path
            logger.warning("未在回复消息中检测到图片组件。")
            return None
        except Exception as e:
            logger.error(f"提取引用图片失败: {e}", exc_info=True)
            return None

    async def _process_image_edit(
        self, event: AstrMessageEvent, prompt: str, image_path: str
    ) -> AsyncGenerator[Comp.MessageComponent, None]:
        """
        处理图片编辑的核心逻辑。
        """
        save_path = None
        try:
            image_data = await self._edit_image_with_retry(prompt, image_path)

            if not image_data:
                yield event.plain_result("图片编辑失败：未能从API获取到图片数据。")
                return

            save_path = os.path.join(self.save_dir, f"{uuid.uuid4()}_edited.png")
            with open(save_path, "wb") as f:
                f.write(image_data)

            logger.info(f"编辑后的图片已临时保存至: {save_path}")
            yield event.chain_result([Comp.Image.fromFileSystem(save_path)])
            logger.info(f"图片编辑完成并发送。")

        except Exception as e:
            logger.error(f"图片编辑过程中发生异常: {e}", exc_info=True)
            yield event.plain_result(f"图片编辑失败，错误信息: {e}")

        finally:
            # 清理临时文件
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    logger.info(f"已清理原始图片临时文件: {image_path}")
                except Exception as e:
                    logger.warning(f"清理原始图片临时文件失败: {e}")

            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已清理编辑后图片临时文件: {save_path}")
                except Exception as e:
                    logger.warning(f"清理编辑后图片临时文件失败: {e}")

    async def _extract_image_url_from_text(self, text_content: str) -> Optional[str]:
        """
        辅助函数：从文本内容中提取图片URL，支持多种格式。
        """
        # 优化正则表达式，使其更简洁和通用
        # 匹配 Markdown, HTML, BBCode 或直接的图片URL
        match = re.search(
            r'(?:!\[.*?\]\((https?://[^\s\)]+)\)|<img[^>]*src=["\'](https?://[^"\'\s]+?)["\']|\[img\](https?://[^\[\]\s]+?)\[/img\]|(https?://\S+\.(?:png|jpg|jpeg|gif|webp)))',
            text_content,
            re.IGNORECASE
        )
        if match:
            # 返回第一个匹配到的URL组
            for i in range(1, match.lastindex + 1):
                url = match.group(i)
                if url:
                    logger.debug(f"提取到图片URL: {url}")
                    return url
        logger.debug("未在文本中提取到图片URL。")
        return None

    async def _download_image_from_url(self, url: str, save_path: str):
        """
        从给定的URL下载图片并保存到指定路径。
        """
        logger.info(f"尝试从URL下载图片: {url} 到 {save_path}")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()  # 检查HTTP状态码，如果不是2xx则抛出异常
                    with open(save_path, "wb") as f:
                        # 异步写入文件，分块读取以处理大文件
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    logger.info(f"图片已成功从URL下载并保存到: {save_path}")
            except aiohttp.ClientError as e:
                logger.error(f"下载图片失败 (aiohttp error): {e}")
                raise
            except Exception as e:
                logger.error(f"下载图片时发生未知错误: {e}")
                raise

    async def _expand_prompt_with_gemini(self, original_prompt: str) -> Optional[str]:
        """
        使用 Gemini 文本模型扩展提示词，只生成正向提示词。
        返回扩展后的正向提示词。
        """
        current_key = self._get_current_api_key()
        if not current_key:
            raise Exception("未配置有效的Gemini API密钥。")

        logger.info(f"尝试扩展提示词（使用密钥：{current_key[:5]}...）")
        return await self._call_gemini_text_model(original_prompt, current_key)

    async def _call_gemini_text_model(self, original_prompt: str, api_key: str) -> Optional[str]:
        """
        实际调用 Gemini 文本模型进行提示词扩展。
        返回扩展后的正向提示词（中文）。
        """
        endpoint = f"{self.api_base_url}/v1beta/models/{self.text_model_name}:generateContent?key={api_key}"
        logger.debug(f"正在向Gemini文本模型发送异步请求: {endpoint}")

        # 使用配置中加载的 prompt_expansion_template，并替换占位符
        llm_prompt = self.prompt_expansion_template.replace("{{original_prompt}}", original_prompt)

        payload = {
            "contents": [{"role": "user", "parts": [{"text": llm_prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "topK": 20,
                "maxOutputTokens": 4096,
            },
        }

        data = await self._send_api_request(endpoint, payload)

        if "candidates" in data and len(data["candidates"]) > 0:
            full_response_text = ""
            for part in data["candidates"][0]["content"]["parts"]:
                if "text" in part:
                    full_response_text += part["text"].strip()

            positive_match = re.search(r'Positive Prompt:\s*(.*)', full_response_text, re.DOTALL)

            positive_prompt = positive_match.group(1).strip() if positive_match else ""

            if positive_prompt:
                logger.info(f"成功获取扩展正向提示词: {positive_prompt}")
                return positive_prompt
        
        logger.warning(f"Gemini文本模型API响应中缺少'candidates'或文本部分，或解析失败: {data}")
        return None # 返回None表示未能成功获取扩展提示词

    async def terminate(self):
        """
        插件卸载时清理临时目录。
        """
        if os.path.exists(self.save_dir):
            try:
                for file in os.listdir(self.save_dir):
                    os.remove(os.path.join(self.save_dir, file))
                os.rmdir(self.save_dir)
                logger.info(f"插件卸载完成：已清理临时目录 {self.save_dir}")
            except Exception as e:
                logger.warning(f"清理临时目录失败: {e}")
        logger.info("Gemini智能绘图插件已成功停用。")

