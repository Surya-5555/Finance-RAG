import re
from typing import List, Dict, Any

class EvidenceValidator:
    """
    Validates generation against the retrieved evidence.
    Specifically checks if any number in the answer actually appears in the cited chunks.
    """
    
    def validate_answer(self, answer: str, contexts: List[Dict[str, Any]]) -> bool:
        """
        Verify that numbers in the answer exist in the provided context chunks.
        """
        # Extract all numbers from the answer (e.g. 1.2, 45, 1,000)
        number_pattern = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b')
        answer_numbers = number_pattern.findall(answer)
        
        if not answer_numbers:
            return True # No numbers to verify
            
        # Combine all context texts
        context_text = " ".join([c["document"].page_content for c in contexts])
        
        # Check if every number is in the context text
        # (This is a simplified exact string match for numbers, in a real system this might need more robust parsing)
        for num in answer_numbers:
            if num not in context_text:
                return False
                
        return True
        
    def get_decline_response(self) -> str:
        return "The information is not available in the uploaded documents."
