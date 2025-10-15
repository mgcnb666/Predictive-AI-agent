"""OpenAI Client Wrapper - Support Calling Models via OpenRouter"""
import os
from typing import Optional, List, Dict, Any
from openai import OpenAI
from loguru import logger


class OpenRouterClient:
    """OpenRouter Client - Using OpenAI SDK"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-2.0-flash-001",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        """
        InitializeOpenRouterClient
        
        Args:
            api_key: OpenRouter API key
            base_url: APIBase URL
            model: Model name
            site_url: WebsiteURL（use Rank）
            site_name: WebsiteName（use Rank）
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.model = model
        self.site_url = site_url or os.getenv("YOUR_SITE_URL")
        self.site_name = site_name or os.getenv("YOUR_SITE_NAME", "Prediction AI Agent")
        
        if not self.api_key:
            raise ValueError("Needextract OPENROUTER_API_KEY")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        logger.info(f"OpenRouterClientInitializeSuccess - Model: {self.model}")
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send DayRequest
        
        Args:
            messages: MessageList
            temperature: TemperatureParameters
            max_tokens: most token 
            **kwargs: OtherParameters
        
        Returns:
            ModelResponseInnercontent
        """
        try:
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                extra_headers["X-Title"] = self.site_name
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers if extra_headers else None,
                **kwargs
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenRouter API useFailed: {e}")
            raise
    
    def chat_with_image(
        self,
        text: str,
        image_url: str,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        SendPackageIncluding image chatDayRequest（Applicable to vision-enabled models）
        
        Args:
            text: TextInnercontent
            image_url: ImageURL
            temperature: TemperatureParameters
            **kwargs: OtherParameters
        
        Returns:
            ModelResponseInnercontent
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
        
        return self.chat(messages=messages, temperature=temperature, **kwargs)
    
    def simple_query(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        SimpleTextQuery
        
        Args:
            query: QueryText
            system_prompt: System promptWord
            temperature: TemperatureParameters
            **kwargs: OtherParameters
        
        Returns:
            ModelResponse
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": query
        })
        
        return self.chat(messages=messages, temperature=temperature, **kwargs)


def create_openrouter_client(
    model: str = "google/gemini-2.0-flash-001",
    **kwargs
) -> OpenRouterClient:
    """
    CreateOpenRouterClientConvenientFunction
    
    Args:
        model: Model name
        **kwargs: OtherParameters
    
    Returns:
        OpenRouterClientInstance
    """
    return OpenRouterClient(model=model, **kwargs)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    client = create_openrouter_client()
    
    print("Testing1: SimpleQuery")
    response = client.simple_query(
        "WorldUpWhat is the fastest land animal？",
        temperature=0.3
    )
    print(f"Answer: {response}\n")
    
    print("Testing2: Sports PredictionQuery")
    response = client.simple_query(
        query="AnalysisManchester UnitedvsLiverpoolmatch，Give win/lossPrediction",
        system_prompt="You are a professionalbody match Analysis ，Good at football match prediction。",
        temperature=0.5
    )
    print(f"Answer: {response}\n")
    
    print("Testing3: Imagerecognize ")
    try:
        response = client.chat_with_image(
            text="What is in this picture？",
            image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )
        print(f"Answer: {response}")
    except Exception as e:
        print(f"Imagerecognize TestingFailed: {e}")

