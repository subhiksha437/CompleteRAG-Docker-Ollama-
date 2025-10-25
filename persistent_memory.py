"""
Persistent Memory System - Remembers across sessions
"""
import os
import json
from datetime import datetime
from typing import Dict, List

class PersistentMemory:
    """Stores user info across all chat sessions"""
    
    def __init__(self, memory_dir="./persistent_memory"):
        self.memory_dir = memory_dir
        self.memory_file = os.path.join(memory_dir, "user_memory.json")
        os.makedirs(memory_dir, exist_ok=True)
        self.data = self._load()
    
    def _load(self) -> Dict:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            "user_info": {},
            "important_facts": [],
            "preferences": {},
            "last_updated": None
        }
    
    def _save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def extract_and_store(self, user_msg: str, assistant_msg: str):
        """Extract important info from conversation"""
        user_lower = user_msg.lower()
        
        # Detect name
        if "my name is" in user_lower or "i'm" in user_lower or "i am" in user_lower:
            for phrase in ["my name is", "i'm", "i am", "call me"]:
                if phrase in user_lower:
                    idx = user_lower.find(phrase) + len(phrase)
                    potential_name = user_msg[idx:].strip().split()[0].strip('.,!?')
                    if len(potential_name) > 1 and potential_name[0].isupper():
                        self.data["user_info"]["name"] = potential_name
                        self._save()
                        break
        
        # Store other facts
        if "i like" in user_lower or "i love" in user_lower:
            self.data["important_facts"].append({
                "fact": user_msg,
                "timestamp": datetime.now().isoformat()
            })
            self._save()
    
    def get_context_str(self) -> str:
        """Get persistent memory as context string"""
        if not self.data["user_info"] and not self.data["important_facts"]:
            return ""
        
        lines = ["### What I Remember About You:"]
        
        if self.data["user_info"].get("name"):
            lines.append(f"- Your name is {self.data['user_info']['name']}")
        
        if self.data["important_facts"]:
            recent_facts = self.data["important_facts"][-5:]
            for fact in recent_facts:
                lines.append(f"- {fact['fact']}")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all persistent memory"""
        self.data = {
            "user_info": {},
            "important_facts": [],
            "preferences": {},
            "last_updated": None
        }
        self._save()
