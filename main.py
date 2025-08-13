import base64
import os
import uuid
import re

import aiohttp
import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register


@register("Gemini_Smart_Canvas", "沐沐沐倾丶", "Gemini智能绘图", "1.0.0",
          description="基于Gemini模型，提供智能文生图和图生图功能，支持提示词自动扩展和固定反向提示词，旨在为用户提供高质量、富有创意的图像生成体验。")
class GeminiImageGenerator(Star):
    # 固定的反向提示词模板，整合了所有提供的英文负面提示
    FIXED_NEGATIVE_PROMPT = "worst quality, normal quality, low quality, low res, blurry, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, pixelated, grainy, color aberration, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, grains, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution, macabre, malformed, mark, misshapen, missing hands, missing legs, mistake, morbid, mutilated, off-screen, outside the picture, poorly drawn feet, printed words, render, repellent, replicate, reproduce, revolting dimensions, script, shortened, sign, split image, squint, storyboard, tiling, trimmed, unfocused, unattractive, unnatural pose, unreal engine, unsightly, written language, bad hands, three hands, three legs, bad arms, missing legs, missing arms, bad face, fused face, cloned face, worst face, out of frame double, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, extra thigh, worst thigh, horn, realistic photo, extra eyes, huge eyes, 2girl, 2boy, amputation, disconnected limbs, poorly rendered hands, bad composition, cartoon, cg, 3d, unreal, animate, cgi, artwork, illustration, 3d render, cinema 4d, artstation, octane render, mutated body parts, painting, oil painting, 2d, sketch, bad photography, bad photo, deviant art, aberrations, abstract, anime, black and white, collapsed, conjoined, creative, drawing, extra windows, harsh lighting, low saturation, multiple levels, overexposed, oversaturated, photoshop, rotten, surreal, twisted, UI, underexposed, unnatural, unrealistic, video game, deformed body features, asymmetrical, unrealistic skin texture, double face, fused hands, too many fingers, deformed hands, ugly eyes, oversized eyes, imperfect eyes, deformed pupils, deformed iris, cross-eyed, asymmetric ears, broken wrist, additional limbs, altered appendages, broken finger, elongated throat, broken hand, combined appendages, broken leg, copied visage, absent limbs, childish, cropped head, cloned head, desiccated, dismembered, disproportionate, cripple"

    def __init__(self, context: Context, config: AstrBotConfig):
        """
        Gemini 图片生成与编辑插件初始化。
        加载配置，设置API密钥，并创建图片临时存储目录。
        """
        super().__init__(context)
        self.config = config

        logger.info(f"Gemini智能绘图插件初始化成功，配置加载: {self.config}")

        # 读取API密钥配置，不再支持多密钥切换，只使用第一个
        self.api_keys = self.config.get("gemini_api_keys", [])

        # 初始化图片保存目录
        plugin_dir = os.path.dirname(__file__)
        self.save_dir = os.path.join(plugin_dir, "temp_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"图片临时存储目录已创建: {self.save_dir}")

        # 初始化API基础URL
        self.api_base_url = self.config.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )
        # 用于提示词扩展的文本生成模型名称
        self.text_model_name = self.config.get("text_model_name", "gemini-2.5-flash")

        if not self.api_keys:
            logger.error("错误：未检测到Gemini API密钥配置，请检查插件设置。")

    def _get_current_api_key(self):
        """
        获取当前使用的 API 密钥。
        由于不再支持多密钥切换，始终返回列表中的第一个密钥。
        """
        if not self.api_keys:
            return None
        return self.api_keys[0] # 始终返回第一个密钥

    # 移除 _switch_next_api_key 方法，不再支持密钥切换

    @filter.command("gemini_image", alias={"文生图"})
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """
        根据文本描述生成图片。
        该方法会尝试扩展用户提示词，并结合固定反向提示词进行图片生成。
        """
        if not self.api_keys:
            yield event.plain_result("错误：插件未配置有效的Gemini API密钥。")
            return

        if not prompt.strip():
            yield event.plain_result(
                "请提供您想要生成的图片描述，例如：/文生图 一只戴帽子的猫在月球上。"
            )
            return

        save_path = None
        prompt_to_send_to_model = prompt # 默认使用原始提示词

        try:
            # 尝试扩展提示词，只获取正向提示词
            expanded_positive_prompt = await self._expand_prompt_with_gemini(prompt)
            
            if expanded_positive_prompt:
                logger.info(f"原始提示词: '{prompt}'")
                logger.info(f"扩展后正向提示词: '{expanded_positive_prompt}'")
                
                # 结合扩展后的正向提示词和固定的反向提示词
                prompt_to_send_to_model = f"{expanded_positive_prompt}, avoid {self.FIXED_NEGATIVE_PROMPT}"
                logger.info(f"发送给图片模型的最终提示词: '{prompt_to_send_to_model}'")
                
                # 不再向用户发送扩展后的提示词内容，只提示正在生成
                yield event.plain_result("正在生成图片，请稍候...")
            else:
                logger.warning("提示词扩展服务异常，将使用原始描述进行图片生成。")
                prompt_to_send_to_model = f"{prompt}, avoid {self.FIXED_NEGATIVE_PROMPT}"
                logger.info(f"发送给图片模型的最终提示词: '{prompt_to_send_to_model}'")
                yield event.plain_result(f"正在生成图片，请稍候...")


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

            # 发送图片
            yield event.chain_result([Image.fromFileSystem(save_path)])
            logger.info(f"图片发送成功。实际使用的提示词（含反向）：'{prompt_to_send_to_model}'")

        except Exception as e:
            logger.error(f"图片生成过程中发生异常: {e}")
            yield event.plain_result(f"图片生成失败，错误信息: {e}")

        finally:
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已清理临时图片文件: {save_path}")
                except Exception as e:
                    logger.warning(f"清理临时图片文件失败: {e}")

    @filter.command("gemini_edit", alias={"图编辑"})
    async def edit_image(self, event: AstrMessageEvent, prompt: str):
        """
        仅支持：引用图片后发送指令编辑图片。
        编辑时只使用用户提供的提示词，并附加固定反向提示词。
        """
        if not self.api_keys:
            yield event.plain_result("错误：插件未配置有效的Gemini API密钥。")
            return

        # 图片提取逻辑
        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result("未检测到引用的图片。请长按图片选择“回复”后，再发送编辑指令。")
            return

        # 编辑图片时，不使用扩展提示词，只使用用户自己的，但添加反向提示词
        prompt_to_send_to_model = f"{prompt}, avoid {self.FIXED_NEGATIVE_PROMPT}"
        logger.info(f"发送给图片编辑模型的最终提示词: '{prompt_to_send_to_model}'")
        yield event.plain_result(f"正在编辑图片，请稍候...")
        # 图片编辑处理
        async for result in self._process_image_edit(event, prompt_to_send_to_model, image_path):
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
    async def edit_image_tool(self, event: AstrMessageEvent, prompt: str):
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
            yield event.plain_result(
                "未检测到引用的图片。请长按图片选择“回复”后，再发送编辑指令。"
            )
            return

        # 编辑图片时，不使用扩展提示词，只使用用户自己的，但添加反向提示词
        prompt_to_send_to_model = f"{prompt}, avoid {self.FIXED_NEGATIVE_PROMPT}"
        logger.info(f"发送给图片编辑模型的最终提示词: '{prompt_to_send_to_model}'")
        yield event.plain_result(f"正在编辑图片，请稍候...")
        async for result in self._process_image_edit(event, prompt_to_send_to_model, image_path):
            yield result

    @filter.llm_tool(name="generate_image",
                     description="根据文本描述生成图片，当你需要生成图片时请使用此工具。")
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """
        LLM工具接口：根据文本描述生成图片。
        Args:
            prompt(string): 图片描述文本（例如：画只猫）
        """
        # 直接调用 generate_image 方法，它已经包含了提示词扩展逻辑
        async for result in self.generate_image(event, prompt):
            yield result

    async def _extract_image_from_reply(self, event: AstrMessageEvent):
        """
        从回复消息中提取图片并返回本地路径。
        """
        try:
            message_components = event.message_obj.message
            reply_component = None
            for comp in message_components:
                if isinstance(comp, Comp.Reply):
                    reply_component = comp
                    logger.debug(f"检测到回复消息（ID：{comp.id}），尝试提取被引用图片。")
                    break

            if not reply_component:
                logger.warning("未检测到回复组件，用户可能未长按图片回复。")
                return None

            # 从回复的chain中提取Image组件
            image_component = None
            for quoted_comp in reply_component.chain:
                if isinstance(quoted_comp, Comp.Image):
                    image_component = quoted_comp
                    logger.debug(f"从回复中提取到图片组件（文件：{image_component.file}）。")
                    break

            if not image_component:
                logger.warning("引用的回复消息中未包含图片组件。")
                return None

            # 获取本地图片路径（自动处理下载/转换）
            image_path = await image_component.convert_to_file_path()
            logger.debug(f"引用的图片已处理为本地路径：{image_path}")
            return image_path

        except Exception as e:
            logger.error(f"提取引用图片失败: {e}", exc_info=True)
            return None

    async def _process_image_edit(
        self, event: AstrMessageEvent, prompt: str, image_path: str
    ):
        """
        处理图片编辑的核心逻辑。
        """
        save_path = None
        try:
            # 调用编辑方法
            image_data = await self._edit_image_manually(prompt, image_path)

            if not image_data:
                yield event.plain_result("图片编辑失败：请检查API密钥或稍后再试。")
                return

            # 保存并发送编辑后的图片
            save_path = os.path.join(self.save_dir, f"{uuid.uuid4()}_edited.png")
            with open(save_path, "wb") as f:
                f.write(image_data)

            yield event.chain_result([Comp.Image.fromFileSystem(save_path)])
            logger.info(f"图片编辑完成并发送。实际使用的提示词（含反向）：'{prompt}'")

        except Exception as e:
            logger.error(f"图片编辑过程中发生异常: {e}")
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

    async def _edit_image_with_retry(self, prompt, image_path):
        """
        带重试逻辑的图片编辑方法。
        由于不再支持多密钥，此方法简化为直接调用编辑函数。
        """
        current_key = self._get_current_api_key()
        if not current_key:
            raise Exception("未配置有效的Gemini API密钥。")

        logger.info(f"尝试编辑图片（使用密钥：{current_key[:5]}...）")
        return await self._edit_image_manually(prompt, image_path, current_key)

    async def _generate_image_with_retry(self, prompt):
        """
        带重试逻辑的图片生成方法。
        由于不再支持多密钥，此方法简化为直接调用生成函数。
        """
        current_key = self._get_current_api_key()
        if not current_key:
            raise Exception("未配置有效的Gemini API密钥。")

        logger.info(f"尝试生成图片（使用密钥：{current_key[:5]}...）")
        return await self._generate_image_manually(prompt, current_key)

    async def _extract_image_url_from_text(self, text_content: str) -> str | None:
        """
        辅助函数：从文本内容中提取图片URL，支持Markdown、HTML、BBCode和直接URL格式。
        优先级：Markdown -> HTML -> BBCode -> 直接URL
        """
        
        # 1. Markdown: !alt text [<sup>1</sup>](url)
        markdown_match = re.search(r'!\[.*?\]\((https?://[^\s\)]+)\)', text_content)
        if markdown_match:
            url = markdown_match.group(1)
            if any(url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                logger.debug(f"提取到Markdown URL: {url}")
                return url

        # 2. HTML <img> tag: <img src="url">
        html_match = re.search(r'<img[^>]*src="\' [<sup>2</sup>](https?://[^"\'\s]+?)["\'][^>]*>', text_content, re.IGNORECASE)
        if html_match:
            url = html_match.group(1)
            if any(url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                logger.debug(f"提取到HTML URL: {url}")
                return url

        # 3. BBCode [img] tag: [img]url[/img]
        bbcode_match = re.search(r'\img\ [<sup>3</sup>](https?://[^\[\]\s]+?)\[/img\]', text_content, re.IGNORECASE)
        if bbcode_match:
            url = bbcode_match.group(1)
            if any(url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                logger.debug(f"提取到BBCode URL: {url}")
                return url

        # 4. Direct URL ending with image extension
        direct_url_match = re.search(r'(https?://\S+\.(?:png|jpg|jpeg|gif|webp))', text_content, re.IGNORECASE)
        if direct_url_match:
            url = direct_url_match.group(1)
            logger.debug(f"提取到直接URL: {url}")
            return url

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

    async def _edit_image_manually(self, prompt, image_path, api_key):
        """
        使用异步请求调用Gemini API编辑图片。
        """
        model_name = "gemini-2.0-flash-preview-image-generation"

        # 修正API地址格式
        base_url = self.api_base_url.strip()
        if not base_url.startswith("https://"):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = (
            f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        )
        logger.debug(f"正在向Gemini图片模型发送异步请求: {endpoint}")

        headers = {"Content-Type": "application/json"}

        # 读取图片并转换为Base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = (
                base64.b64encode(image_bytes)
                .decode("utf-8")
                .replace("\n", "")
                .replace("\r", "")
            )

        # 构建请求参数
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": image_base64}}
                    ],
                },
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 2048, # 图片生成模型通常不需要太高的maxOutputTokens
            },
        }

        # 异步发送请求
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url=endpoint, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(
                            f"Gemini图片编辑API请求失败: HTTP {response.status}, 响应: {response_text}"
                        )
                        response.raise_for_status()

                    data = await response.json()
                    logger.debug(f"Gemini图片编辑API响应数据: {data}")

            except Exception as e:
                logger.error(f"异步图片编辑请求失败: {e}")
                raise

        # 解析图片数据或URL
        image_data = None
        image_url = None

        if "candidates" in data and len(data["candidates"]) > 0:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    base64_str = (
                        part["inlineData"]["data"].replace("\n", "").replace("\r", "")
                    )
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
            temp_download_path = os.path.join(self.save_dir, f"downloaded_edit_{uuid.uuid4()}.png")
            try:
                await self._download_image_from_url(image_url, temp_download_path)
                with open(temp_download_path, "rb") as f:
                    downloaded_image_bytes = f.read()
                logger.debug(f"成功从URL下载图片: {image_url}")
                return downloaded_image_bytes
            except Exception as download_e:
                logger.error(f"从URL下载图片失败 {image_url}: {download_e}")
                raise Exception(f"图片编辑成功，但从URL下载图片失败: {download_e}")
            finally:
                if os.path.exists(temp_download_path):
                    os.remove(temp_download_path)
                    logger.debug(f"已清理临时下载文件: {temp_download_path}")
        else:
            raise Exception("图片编辑成功，但未能获取到图片数据或URL。")

    async def _generate_image_manually(self, prompt, api_key):
        """
        使用异步请求调用Gemini API生成图片。
        """
        model_name = "gemini-2.0-flash-preview-image-generation"

        base_url = self.api_base_url.strip()
        if not base_url.startswith("https://"):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = (
            f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        )
        logger.debug(f"正在向Gemini图片模型发送异步请求: {endpoint}")

        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 2048, # 图片生成模型通常不需要太高的maxOutputTokens
            },
        }

        # 异步发送请求
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url=endpoint, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(
                            f"Gemini图片生成API请求失败: HTTP {response.status}, 响应: {response_text}"
                        )
                        response.raise_for_status()

                    data = await response.json()
                    logger.debug(f"Gemini图片生成API响应数据: {data}")

            except Exception as e:
                logger.error(f"异步图片生成请求失败: {e}")
                raise

        # 解析图片数据或URL
        image_data = None
        image_url = None

        if "candidates" in data and len(data["candidates"]) > 0:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    base64_str = (
                        part["inlineData"]["data"].replace("\n", "").replace("\r", "")
                    )
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
            temp_download_path = os.path.join(self.save_dir, f"downloaded_gen_{uuid.uuid4()}.png")
            try:
                await self._download_image_from_url(image_url, temp_download_path)
                with open(temp_download_path, "rb") as f:
                    downloaded_image_bytes = f.read()
                logger.debug(f"成功从URL下载图片: {image_url}")
                return downloaded_image_bytes
            except Exception as download_e:
                logger.error(f"从URL下载图片失败 {image_url}: {download_e}")
                raise Exception(f"图片生成成功，但从URL下载图片失败: {download_e}")
            finally:
                if os.path.exists(temp_download_path):
                    os.remove(temp_download_path)
                    logger.debug(f"已清理临时下载文件: {temp_download_path}")
        else:
            raise Exception("图片生成成功，但未能获取到图片数据或URL。")

    async def _expand_prompt_with_gemini(self, original_prompt: str) -> str | None:
        """
        使用 Gemini 文本模型扩展提示词，只生成正向提示词。
        返回扩展后的正向提示词。
        """
        current_key = self._get_current_api_key()
        if not current_key:
            raise Exception("未配置有效的Gemini API密钥。")

        logger.info(f"尝试扩展提示词（使用密钥：{current_key[:5]}...）")
        return await self._call_gemini_text_model(original_prompt, current_key)

    async def _call_gemini_text_model(self, original_prompt: str, api_key: str) -> str:
        """
        实际调用 Gemini 文本模型进行提示词扩展。
        返回扩展后的正向提示词（中文）。
        """
        base_url = self.api_base_url.strip()
        if not base_url.startswith("https://"):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = (
            f"{base_url}/v1beta/models/{self.text_model_name}:generateContent?key={api_key}"
        )
        logger.debug(f"正在向Gemini文本模型发送异步请求: {endpoint}")

        headers = {"Content-Type": "application/json"}

        # 构建给LLM的提示词，引导其生成详细的中文正向提示词
        llm_prompt = f"""你是一个专业的图片生成提示词扩展助手。请根据用户提供的简短描述，将其扩展为详细、富有创意且包含丰富细节的图片生成**正向提示词 (Positive Prompt)**。

请严格按照以下格式输出，不要包含任何额外说明或对话内容：
Positive Prompt: [这里是详细的图片生成正向提示词]

请在扩展正向提示词时考虑以下方面，使其适用于gemini-2.0-flash-preview-image-generation模型：
1.  **主体细节**: 描述主体的具体特征、动作、表情、服装、材质、纹理等。
2.  **环境/背景**: 场景的地点、时间、天气、光线（如柔和光、强对比光）、氛围、周围物体、景深等。
3.  **风格/艺术性**: 艺术风格（如赛博朋克、印象派、写实、动漫风格、油画、水彩）、画质（如8K、超高清）、光影效果、色彩搭配、渲染方式（如3D渲染、数字绘画）。
4.  **构图/视角**: 构图方式（如特写、全身照、广角、全景、对称构图）、视角（如俯视、仰视、平视）、景深。
5.  **情绪/主题**: 图片想要表达的情绪、故事或主题。

原始描述: '{original_prompt}'"""

        payload = {
            "contents": [{"role": "user", "parts": [{"text": llm_prompt}]}],
            "generationConfig": {
                "temperature": 0.7, # 适当降低温度以获得更稳定的结果
                "topP": 0.9,
                "topK": 20,
                "maxOutputTokens": 4096, # 增加最大输出令牌数量以确保完整扩展
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=endpoint, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(
                        f"Gemini文本模型API请求失败: HTTP {response.status}, 响应: {response_text}"
                    )
                    response.raise_for_status()

                data = await response.json()
                logger.debug(f"Gemini文本模型API响应数据: {data}")

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
                raise Exception("文本模型未能返回有效的扩展提示词。")

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
