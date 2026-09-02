import re

class SmartRouter:
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