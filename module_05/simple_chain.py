import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_gigachat.chat_models import GigaChat

from rag_stub import answer as rag_answer
# from rag_adapter import answer as rag_answer

load_dotenv(Path(__file__).parent.parent / ".env")

CATEGORIES = ["регламент", "доступ", "инцидент", "документация"]

REQUESTS = json.loads(
    (Path(__file__).parent / "data" / "requests.json").read_text(encoding="utf-8")
)["requests"]


def make_llm() -> GigaChat:
    """Модель как звено цепочки. Параметры — как в Модулях 1-2."""
    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
        model="GigaChat-2",  # модель ВСЕГДА явно: алиасов «по умолчанию» на новом API нет
        temperature=0.0,  # классификация и ответы должны быть повторяемыми
        max_tokens=300,
    )


# ====================================================================
#  ЗВЕНО 1. Классификация обращения
# ====================================================================


CLASSIFY_PROMPT = """Определи категорию обращения сотрудника промышленного предприятия.
Ответь ОДНИМ словом из списка: регламент, доступ, инцидент, документация.

Примеры:
Обращение: "Станок ЧПУ выдаёт ошибку шпинделя и встал" -> инцидент
Обращение: "Заведите учётную запись новому технологу" -> доступ
Обращение: "В какой срок нужно оформить карточку инцидента после остановки?" -> регламент
Обращение: "Нужен паспорт на питательный электронасос №4" -> документация
Обращение: "Задвижка на линии подпидки не работает" -> инцидент
Обращение: "Кто должен расписываться в журнале охраны труда" -> регламент

Обращение: "{question}" ->"""


def pick_category(raw: str) -> str:
    """Достаёт категорию из сырого ответа модели."""
    text = raw.lower()
    if "категория" in text:
        text = text.split("категория", 1)[1]

    found = [(text.find(c), c) for c in CATEGORIES if c in text]
    if not found:
        return "не определено"
    found.sort()
    return found[0][1]


def build_classify_chain(model):
    """Цепочка классификации: промпт -> модель -> строка -> категория."""

    return (
        ChatPromptTemplate.from_template(CLASSIFY_PROMPT)
        | model
        | StrOutputParser()
        | RunnableLambda(pick_category)
    )


# ====================================================================
#  ЗВЕНО 2. Ветки ответа
# ====================================================================

ANSWER_PROMPTS = {
    "инцидент": (
        "Ты дежурный диспетчер промышленного предприятия. Поступило сообщение "
        'об инциденте:\n"{question}"\n'
        "Дай 2-3 коротких первых шага (безопасность людей, остановка оборудования, "
        "кого известить). Не выдумывай технических деталей, которых нет в сообщении. "
        "В конце сообщи, что заявка передана дежурному инженеру."
    ),
    "доступ": (
        "Ты ассистент службы ИТ промышленного предприятия. Поступила заявка "
        'на доступ:\n"{question}"\n'
        "Вежливо и коротко ответь: заявка зарегистрирована, доступ выдаётся после "
        "согласования руководителем подразделения. Уточни, какие данные нужны "
        "(ФИО, подразделение, система), если их нет в заявке."
    ),
}

# какие категории закрывает база знаний, а какие — генерация по промпту
RAG_CATEGORIES = ("регламент", "документация")


def rag_node(payload: dict) -> dict:
    """Ветка поиска: вопрос уходит в RAG-звено, ответ — с источниками."""
    result = rag_answer(payload["question"])
    return {
        "category": payload["category"],
        "answer": result["answer"],
        "sources": result["sources"],
        "action": "ответ" if result["sources"] else "специалисту",
    }


def generate_node(payload: dict, model) -> dict:
    """Ветка генерации: ответ по промпту категории. Источников у неё нет."""
    prompt = ChatPromptTemplate.from_template(ANSWER_PROMPTS[payload["category"]])
    chain = prompt | model | StrOutputParser()
    return {
        "category": payload["category"],
        "answer": chain.invoke({"question": payload["question"]}),
        "sources": [],
        "action": "специалисту",  # инцидент и доступ всегда закрывает человек
    }


def route_answer(payload: dict, model) -> dict:
    """Выбор ветки по категории."""
    category = payload["category"]
    if category in RAG_CATEGORIES:
        return rag_node(payload)
    if category in ANSWER_PROMPTS:
        return generate_node(payload, model)
    return {
        "category": category,
        "answer": "Не удалось определить тип обращения - передаю оператору.",
        "sources": [],
        "action": "специалисту",
    }


def build_full_chain(model):
    """Полная цепочка сценария.

    Первый шаг — словарь из двух ветвей, которые LCEL выполняет над одним
    и тем же входом: classify-цепочка даёт категорию, лямбда прокидывает
    исходный вопрос дальше. Второй шаг — выбор ветки ответа.
    """
    return {
        "category": build_classify_chain(model),
        "question": RunnableLambda(lambda x: x["question"]),
    } | RunnableLambda(lambda payload: route_answer(payload, model))


def main():
    llm = make_llm()
    chain = build_full_chain(llm)

    for req in REQUESTS:
        result = chain.invoke({"question": req["text"]})
        print("=" * 70)
        print(f"Обращение : {req['text']}")
        print(f"Категория : {result['category']}")
        print(f"Действие  : {result['action']}")
        print(f"Ответ     : {result['answer']}")
        if result["sources"]:
            print(f"Источники : {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()