"""Context Manager - Supports Context Sharing Between Agents"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
from loguru import logger


class ContextManager:
    """ContextManagementer - ManagementConversation HistoryandUserPreference"""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        InitializeContextManagementer
        
        Args:
            session_id: SessionID，Used to distinguish different users/Session
        """
        self.session_id = session_id or self._generate_session_id()
        
        self.conversation_history: List[Dict[str, Any]] = []
        
        self.preferences: Dict[str, Any] = {
            "default_location": None,  
            "timezone": "Asia/Shanghai",  
            "language": "zh",  
            "favorite_teams": [],  
        }
        
        self.context_vars: Dict[str, Any] = {}
        
        self.recent_params: Dict[str, Any] = {}
        
        self.domain_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info(f"InitializeContextManagementer - Session: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """GenerateSessionID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        AddConversationMessage
        
        Args:
            role: Role (user/assistant/system)
            content: MessageInnercontent
            metadata:  Outer Data
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(message)
        logger.debug(f"AddMessage: {role} - {content[:50]}...")
    
    def add_prediction(self, domain: str, params: Dict[str, Any], result: Dict[str, Any]):
        """
        RecordPrediction Result
        
        Args:
            domain: Prediction domain
            params: Prediction parameters
            result: Prediction Result
        """
        record = {
            "domain": domain,
            "params": params,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        self.domain_history[domain].append(record)
        self.recent_params.update(params)
        
        logger.info(f"RecordPrediction - {domain}: {params}")
    
    def get_context_for_domain(self, domain: str) -> Dict[str, Any]:
        """
        GetSpecificDomainContext
        
        Args:
            domain: DomainName
        
        Returns:
            DomainContext
        """
        return {
            "history": self.domain_history.get(domain, []),
            "recent_params": self.recent_params,
            "preferences": self.preferences,
            "conversation": self.conversation_history[-5:],  
        }
    
    def smart_complete_params(self, domain: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Smart parameter completion
        
        Args:
            domain: Domain
            params: whenBeforeParameters
        
        Returns:
            CompleteAfterParameters
        """
        completed = params.copy()
        
        if domain == "weather":
            if not completed.get("location"):
                completed["location"] = (
                    self.recent_params.get("location") or
                    self.preferences.get("default_location") or
                    "Beijing"
                )
                logger.info(f"CompleteLocation: {completed['location']}")
        
        elif domain == "sports":
            if completed.get("team1") and not completed.get("team2"):
                for record in reversed(self.domain_history.get("sports", [])):
                    if record["params"].get("team1") == completed["team1"]:
                        completed["team2"] = record["params"].get("team2")
                        logger.info(f"CompleteOpponent: {completed['team2']}")
                        break
        
        return completed
    
    def set_preference(self, key: str, value: Any):
        """SettingUserPreference"""
        self.preferences[key] = value
        logger.info(f"set preference: {key} = {value}")
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """GetUserPreference"""
        return self.preferences.get(key, default)
    
    def set_context_var(self, key: str, value: Any):
        """SettingContextVariable"""
        self.context_vars[key] = value
        logger.debug(f"SettingVariable: {key} = {value}")
    
    def get_context_var(self, key: str, default: Any = None) -> Any:
        """GetContextVariable"""
        return self.context_vars.get(key, default)
    
    def get_conversation_context(self, max_turns: int = 10) -> str:
        """
        GetConversationContextSummary
        
        Args:
            max_turns: most ConversationRound
        
        Returns:
            ContextSummaryText
        """
        recent = self.conversation_history[-max_turns*2:]
        
        context_lines = []
        for msg in recent:
            role = msg["role"]
            content = msg["content"]
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)
    
    def summarize(self) -> Dict[str, Any]:
        """GenerateContextSummary"""
        return {
            "session_id": self.session_id,
            "conversation_count": len(self.conversation_history),
            "predictions": {
                domain: len(records)
                for domain, records in self.domain_history.items()
            },
            "recent_params": self.recent_params,
            "preferences": self.preferences,
        }
    
    def save_to_file(self, filepath: str):
        """SaveContextto file"""
        data = {
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "preferences": self.preferences,
            "context_vars": self.context_vars,
            "recent_params": self.recent_params,
            "domain_history": dict(self.domain_history),
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Context Saveto: {filepath}")
    
    def load_from_file(self, filepath: str):
        """from itemLoadContext"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.session_id = data.get("session_id", self.session_id)
        self.conversation_history = data.get("conversation_history", [])
        self.preferences = data.get("preferences", self.preferences)
        self.context_vars = data.get("context_vars", {})
        self.recent_params = data.get("recent_params", {})
        
        domain_history_data = data.get("domain_history", {})
        self.domain_history = defaultdict(list)
        for domain, records in domain_history_data.items():
            self.domain_history[domain] = records
        
        logger.info(f"from itemLoadContext: {filepath}")
    
    def clear(self):
        """ClearAllContext"""
        self.conversation_history.clear()
        self.context_vars.clear()
        self.recent_params.clear()
        self.domain_history.clear()
        logger.info("ContextCleared")


class SharedContextManager:
    """ShareContextManagementer - Supports multiple agent sharingContext"""
    
    _instances: Dict[str, ContextManager] = {}
    
    @classmethod
    def get_context(cls, session_id: str) -> ContextManager:
        """
        GetorCreateSessionContext
        
        Args:
            session_id: SessionID
        
        Returns:
            ContextManagementerInstance
        """
        if session_id not in cls._instances:
            cls._instances[session_id] = ContextManager(session_id)
            logger.info(f"Create SessionContext: {session_id}")
        
        return cls._instances[session_id]
    
    @classmethod
    def remove_context(cls, session_id: str):
        """DeleteSessionContext"""
        if session_id in cls._instances:
            del cls._instances[session_id]
            logger.info(f"DeleteSessionContext: {session_id}")
    
    @classmethod
    def list_sessions(cls) -> List[str]:
        """List all active sessions"""
        return list(cls._instances.keys())
    
    @classmethod
    def clear_all(cls):
        """ClearAllSessionContext"""
        cls._instances.clear()
        logger.info("ClearAllSessionContext")


_default_context = None


def get_default_context() -> ContextManager:
    """GetDefaultContextManagementer"""
    global _default_context
    if _default_context is None:
        _default_context = ContextManager(session_id="default")
    return _default_context


if __name__ == "__main__":
    print("="*70)
    print("TestingContextManagementer")
    print("="*70)
    
    ctx = ContextManager(session_id="test-001")
    
    ctx.add_message("user", "Predict tomorrow's weather in Beijing")
    ctx.add_message("assistant", "TomorrowDayBeijingsunny，15-22 ")
    
    ctx.add_prediction(
        domain="weather",
        params={"location": "Beijing", "date": "2025-10-16"},
        result={"forecast": {"weather_condition": "sunny"}}
    )
    
    print("\nContextSummary:")
    print(json.dumps(ctx.summarize(), indent=2, ensure_ascii=False))
    
    print("\nTestingSmartComplete:")
    incomplete_params = {"date": "2025-10-17"}
    completed = ctx.smart_complete_params("weather", incomplete_params)
    print(f"CompleteBefore: {incomplete_params}")
    print(f"CompleteAfter: {completed}")
    
    print("\nTestingShareContext:")
    ctx1 = SharedContextManager.get_context("user-001")
    ctx1.set_preference("default_location", "Uphai")
    
    ctx2 = SharedContextManager.get_context("user-001")
    print(f"PreferenceShare: {ctx2.get_preference('default_location')}")
    
    print(f"\nActiveSession: {SharedContextManager.list_sessions()}")

