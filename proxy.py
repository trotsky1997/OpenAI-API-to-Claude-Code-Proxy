#!/usr/bin/env python3
"""
本地代理服务器：将 Anthropic /v1/messages 请求转换为 OpenAI /v1/chat/completions 请求
并将 OpenAI 响应转换回 Anthropic 格式

使用方法:
1. 启动代理服务器: python proxy_server.py
2. 修改环境变量: ANTHROPIC_BASE_URL=http://localhost:8000
3. 运行测试: python test_claude_agent.py
"""

import os
import json
import uuid
import asyncio
from typing import Any, Dict, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.parse
import urllib.error

# 配置
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))
TARGET_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://xxxxx:xxx").rstrip("/")
TARGET_API_KEY = os.getenv("OPENAI_API_KEY", "")


def convert_anthropic_to_openai_request(anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 Anthropic /v1/messages 请求转换为 OpenAI /v1/chat/completions 请求
    
    Anthropic 格式:
    {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Hello!"
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi!"},
                    {"type": "tool_use", "id": "call_123", "name": "execute_sql", "input": {...}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_123", "content": "result"}
                ]
            }
        ],
        "tools": [...]
    }
    
    OpenAI 格式:
    {
        "model": "gpt-4",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!", "tool_calls": [...]},
            {"role": "tool", "content": "result", "tool_call_id": "call_123"}
        ],
        "tools": [...]
    }
    """
    openai_request = {
        "model": anthropic_request.get("model", "gpt-4"),
        "max_tokens": anthropic_request.get("max_tokens", 1024),
        "temperature": anthropic_request.get("temperature"),
        "stream": anthropic_request.get("stream", False),
    }
    
    # 转换 messages
    messages = []
    anthropic_messages = anthropic_request.get("messages", [])
    
    if not anthropic_messages:
        print(f"[Proxy] Warning: No messages in request")
        return openai_request
    
    for msg in anthropic_messages:
        role = msg.get("role")
        content = msg.get("content")
        
        if role == "system":
            # OpenAI 使用 system 消息
            if isinstance(content, str):
                messages.append({"role": "system", "content": content})
            elif isinstance(content, list):
                # 提取文本内容
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                if text_parts:
                    messages.append({"role": "system", "content": "\n".join(text_parts)})
        elif role == "user":
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # 处理 tool_result
                tool_results = [item for item in content if isinstance(item, dict) and item.get("type") == "tool_result"]
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                
                if tool_results:
                    # OpenAI 使用多个 tool 消息
                    for tool_result in tool_results:
                        tool_content = tool_result.get("content", "")
                        if not isinstance(tool_content, str):
                            tool_content = json.dumps(tool_content) if tool_content else ""
                        messages.append({
                            "role": "tool",
                            "content": tool_content,
                            "tool_call_id": tool_result.get("tool_use_id", ""),
                        })
                
                if text_parts:
                    messages.append({"role": "user", "content": "\n".join(text_parts)})
            elif content is None:
                # 空内容
                messages.append({"role": "user", "content": ""})
        elif role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                # 提取文本和 tool_use
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                tool_uses = [item for item in content if isinstance(item, dict) and item.get("type") == "tool_use"]
                
                assistant_msg = {"role": "assistant"}
                
                if text_parts:
                    assistant_msg["content"] = "\n".join(text_parts)
                elif not tool_uses:
                    assistant_msg["content"] = ""
                
                if tool_uses:
                    # 转换为 tool_calls
                    tool_calls = []
                    for tool_use in tool_uses:
                        tool_name = tool_use.get("name", "")
                        tool_input = tool_use.get("input", {})
                        if tool_name:  # 只添加有效的工具调用
                            tool_calls.append({
                                "id": tool_use.get("id", f"call_{uuid.uuid4().hex[:16]}"),
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_input) if not isinstance(tool_input, str) else tool_input,
                                }
                            })
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                
                messages.append(assistant_msg)
            elif content is None:
                # 空内容
                messages.append({"role": "assistant", "content": ""})
    
    openai_request["messages"] = messages
    
    # 转换 tools
    anthropic_tools = anthropic_request.get("tools", [])
    if anthropic_tools:
        openai_tools = []
        for tool in anthropic_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                }
            }
            openai_tools.append(openai_tool)
        openai_request["tools"] = openai_tools
    
    return openai_request


def convert_openai_to_anthropic_response(openai_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 OpenAI /v1/chat/completions 响应转换为 Anthropic /v1/messages 响应
    
    OpenAI 格式:
    {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Hello!",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "execute_sql",
                        "arguments": "{\"sql_query\": \"SELECT * FROM table\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {...}
    }
    
    Anthropic 格式:
    {
        "id": "msg_xxx",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Hello!"},
            {"type": "tool_use", "id": "call_123", "name": "execute_sql", "input": {"sql_query": "SELECT * FROM table"}}
        ],
        "model": "...",
        "stop_reason": "tool_use",
        "usage": {...}
    }
    """
    if not openai_response.get("choices"):
        return openai_response
    
    choice = openai_response["choices"][0]
    message = choice.get("message", {})
    
    # 构建 content 数组
    content = []
    
    # 1. 处理文本内容
    text_content = message.get("content")
    if text_content:
        content.append({"type": "text", "text": text_content})
    
    # 2. 处理 tool_calls
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        tool_calls = []
    elif not isinstance(tool_calls, list):
        tool_calls = []
    
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        
        tool_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:16]}")
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            continue
        
        tool_name = function.get("name", "")
        if not tool_name:
            continue
        
        # 解析 arguments
        arguments_str = function.get("arguments", "{}")
        try:
            if isinstance(arguments_str, dict):
                parsed_input = arguments_str
            elif isinstance(arguments_str, str):
                parsed_input = json.loads(arguments_str)
            else:
                parsed_input = {}
        except (json.JSONDecodeError, TypeError):
            parsed_input = {"text": str(arguments_str)}
        
        content.append({
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": parsed_input
        })
    
    # 3. 处理 finish_reason
    finish_reason = choice.get("finish_reason", "")
    stop_reason = "tool_use" if finish_reason == "tool_calls" else finish_reason
    
    # 4. 转换 usage
    usage = openai_response.get("usage", {})
    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }
    
    # 5. 构建 Anthropic 响应
    anthropic_response = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4().hex[:16]}"),
        "type": "message",
        "role": "assistant",
        "content": content if content else [{"type": "text", "text": ""}],
        "model": openai_response.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage,
    }
    
    return anthropic_response


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """处理 POST 请求"""
        # 支持 /v1/messages 路径（可能带查询参数）
        path_without_query = self.path.split("?")[0]
        print(f"[Proxy] Received POST request to: {self.path} (path: {path_without_query})")
        if path_without_query == "/v1/messages" or path_without_query.endswith("/v1/messages"):
            self.handle_anthropic_request()
        else:
            print(f"[Proxy] 404: Path not matched: {self.path}")
            self.send_error(404, f"Not Found: {self.path}")
    
    def do_GET(self):
        """处理 GET 请求（健康检查）"""
        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "proxy": "running"}).encode("utf-8"))
        else:
            self.send_error(404, f"Not Found: {self.path}")
    
    def handle_anthropic_request(self):
        """处理 Anthropic /v1/messages 请求"""
        try:
            # 读取请求体
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Empty request body")
                return
            
            request_body = self.rfile.read(content_length)
            try:
                anthropic_request = json.loads(request_body.decode("utf-8"))
            except json.JSONDecodeError as e:
                print(f"[Proxy] Invalid JSON: {e}")
                self.send_error(400, f"Invalid JSON: {e}")
                return
            
            print(f"[Proxy] Received Anthropic request: model={anthropic_request.get('model')}, messages={len(anthropic_request.get('messages', []))}")
            
            # 转换为 OpenAI 格式
            try:
                openai_request = convert_anthropic_to_openai_request(anthropic_request)
                print(f"[Proxy] Converted to OpenAI request: {len(openai_request.get('messages', []))} messages")
            except Exception as e:
                print(f"[Proxy] Error converting request: {e}")
                import traceback
                traceback.print_exc()
                self.send_error(500, f"Conversion error: {e}")
                return
            
            # 转发到目标服务器
            target_url = f"{TARGET_BASE_URL}/v1/chat/completions"
            
            # 准备请求数据
            request_data = json.dumps(openai_request).encode("utf-8")
            print(f"[Proxy] Forwarding to: {target_url}")
            print(f"[Proxy] Request size: {len(request_data)} bytes")
            
            req = urllib.request.Request(
                target_url,
                data=request_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {TARGET_API_KEY}",
                }
            )
            
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    response_body = response.read().decode("utf-8")
                    if not response_body:
                        print(f"[Proxy] Empty response from target server")
                        self.send_error(500, "Empty response from target server")
                        return
                    
                    try:
                        openai_response = json.loads(response_body)
                    except json.JSONDecodeError as e:
                        print(f"[Proxy] Invalid JSON in response: {e}")
                        print(f"[Proxy] Response body (first 500 chars): {response_body[:500]}")
                        self.send_error(500, f"Invalid JSON response: {e}")
                        return
                    
                    print(f"[Proxy] Received OpenAI response: finish_reason={openai_response.get('choices', [{}])[0].get('finish_reason', 'N/A')}")
                    
                    # 转换为 Anthropic 格式
                    try:
                        anthropic_response = convert_openai_to_anthropic_response(openai_response)
                        print(f"[Proxy] Converted to Anthropic response: stop_reason={anthropic_response.get('stop_reason')}")
                    except Exception as e:
                        print(f"[Proxy] Error converting response: {e}")
                        import traceback
                        traceback.print_exc()
                        self.send_error(500, f"Conversion error: {e}")
                        return
                    
                    # 返回响应
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(anthropic_response).encode("utf-8"))
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8")
                print(f"[Proxy] Error from target server: {e.code} - {error_body[:200]}")
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    # 尝试解析错误响应并转换格式
                    error_json = json.loads(error_body)
                    self.wfile.write(json.dumps(error_json).encode("utf-8"))
                except:
                    self.wfile.write(error_body.encode("utf-8"))
            except Exception as e:
                print(f"[Proxy] Error forwarding request: {e}")
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": str(e), "type": "proxy_error"}}).encode("utf-8"))
        
        except json.JSONDecodeError as e:
            print(f"[Proxy] Invalid JSON in request: {e}")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}}).encode("utf-8"))
        except Exception as e:
            print(f"[Proxy] Error processing request: {e}")
            import traceback
            traceback.print_exc()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": str(e), "type": "proxy_error"}}).encode("utf-8"))
    
    def log_message(self, format, *args):
        """禁用默认日志"""
        pass


def main():
    """启动代理服务器"""
    server = HTTPServer(("localhost", PROXY_PORT), ProxyHandler)
    print(f"[Proxy] Starting proxy server on http://localhost:{PROXY_PORT}")
    print(f"[Proxy] Forwarding to: {TARGET_BASE_URL}/v1/chat/completions")
    print(f"[Proxy] Set ANTHROPIC_BASE_URL=http://localhost:{PROXY_PORT}")
    print(f"[Proxy] Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Proxy] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
