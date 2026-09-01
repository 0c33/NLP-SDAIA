"""
Smart Router for Bayan Pipeline.
Routes user intent to the correct NLP task without calling heavy models.
"""
import re

class SmartRouter:
    """Pure Python agentic router to save compute.
    
    FIX: Arabic interrogatives like 'من' (who/from) and 'ما' (what/negation) 
    are highly ambiguous. Routing now requires an explicit question mark 
    before considering it a QA intent.
    """
    QUESTION_MARK = re.compile(r"[؟?]")
    SEARCH_HINT = re.compile(r"ابحث|وثائق|مستندات|search|find|documents", re.IGNORECASE)
    NER_HINT = re.compile(r"استخرج|كيانات|أسماء|extract|entities", re.IGNORECASE)

    @staticmethod
    def route(text: str) -> str:
        if SmartRouter.QUESTION_MARK.search(text):
            return "qa"
        if SmartRouter.SEARCH_HINT.search(text):
            return "search"
        if SmartRouter.NER_HINT.search(text):
            return "ner"
        return "classification"