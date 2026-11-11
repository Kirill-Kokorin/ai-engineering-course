import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import MCPServer

BASE_DIR = Path(__file__).parent
EQUIPMENT_PATH = BASE_DIR / "data" / "equipment_registry.json"
INCIDENTS_PATH = BASE_DIR / "data" / "incidents.json"
AUDIT_LOG_PATH = BASE_DIR / "data" / "audit_log.jsonl"

# Роли, которым разрешено создавать инциденты (write-инструмент).
# "viewer" — только чтение, дефолт нарочно "безопасный": если вызывающий
# не передал роль явно, он получит отказ на create_incident, а не заявку.
ALLOWED_TO_CREATE_INCIDENT = {"engineer", "admin"}
ALLOWED_PRIORITIES = {"low", "medium", "high", "critical"}



# В боевой системе ПромТеха это оборудование уже стоит на учёте, но в
# нашем учебном реестре (data/equipment_registry.json) его пока нет —
# заполни карточку по описанию ниже и запись подключится к общему
# реестру при старте сервера.
#
# Описание для карточки:
#   Транспортёрная лента ТЛ-9, установлена в цехе дробления,
#   сейчас в работе, дата последнего технического обслуживания — 10
#   февраля 2026 года (формат даты — как в остальных записях реестра,
#   "YYYY-MM-DD").

NEW_EQUIPMENT = {
    "id": "CONV-09",
    "name": "Транспортёрная лента ТЛ-9",
    "type": "Транспортная лента",
    "location": "Цех дробления",
    "status": "В работе",
    "last_service_date": "2026-02-10",
}


# ====================================================================
#  ИНФРАСТРУКТУРА (дано, менять не нужно)
# ====================================================================

class CorporateSystemUnavailable(Exception):
    """Имитация отказа боевой системы ПромТеха (сценарий outage, занятие 12)."""


# Переключатель сбоя «боевой» системы. Сам по себе действует только внутри
# того процесса, где его выставили. mcp_client_demo.py поднимает сервер
# отдельным процессом — поэтому shift_review_m12.py включает сбой через
# файл data/simulate_outage.flag, а сервер читает его при каждом вызове.
SIMULATE_OUTAGE = False
OUTAGE_FLAG_PATH = BASE_DIR / "data" / "simulate_outage.flag"


def outage_is_on() -> bool:
    return SIMULATE_OUTAGE or OUTAGE_FLAG_PATH.exists()


def set_simulate_outage(on: bool) -> None:
    """Включает/выключает сбой и для текущего процесса, и для MCP-сервера в хосте."""
    global SIMULATE_OUTAGE
    SIMULATE_OUTAGE = bool(on)
    if on:
        OUTAGE_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTAGE_FLAG_PATH.write_text("1", encoding="utf-8")
    elif OUTAGE_FLAG_PATH.exists():
        OUTAGE_FLAG_PATH.unlink()


def _load_equipment() -> dict:
    """Читает реестр оборудования и подмешивает учебную запись (TODO 1)."""
    registry = json.loads(EQUIPMENT_PATH.read_text(encoding="utf-8-sig"))
    by_id = {item["id"]: item for item in registry["equipment"]}
    by_id[NEW_EQUIPMENT["id"]] = NEW_EQUIPMENT
    return by_id


EQUIPMENT = _load_equipment()


def _load_incidents() -> list:
    if not INCIDENTS_PATH.exists():
        return []
    return json.loads(INCIDENTS_PATH.read_text(encoding="utf-8-sig"))


def _save_incident(incident: dict) -> None:
    incidents = _load_incidents()
    incidents.append(incident)
    INCIDENTS_PATH.write_text(
        json.dumps(incidents, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def log_action(action: str, params: dict, result: dict) -> None:
    """Журналирует каждый вызов инструмента: кто, что и с каким итогом.

    В промышленном контуре это не «для галочки» — по такому логу потом
    разбирают инциденты и проверяют, что ИИ-ассистент не вышел за рамки
    разрешённых действий. Формат — JSON Lines: одна запись — одна строка,
    удобно дописывать и удобно потом построчно прочитать при аудите.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "params": params,
        "ok": bool(result.get("ok")),
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _simulate_corporate_lookup(equipment_id: str) -> None:
    """Заглушка вызова реальной корпоративной системы ПромТеха.

    В учебном проекте статус и так лежит в локальном реестре, но в реальном
    MCP-сервере get_equipment_status обычно дергает внутренний API/БД —
    и этот вызов может упасть (сеть, сервис не отвечает, права не те).
    Этой функцией моделируем такой отказ: переменная SIMULATE_OUTAGE
    или файл data/simulate_outage.flag (его ставит shift_review_m12.py).
    """
    if outage_is_on():
        raise CorporateSystemUnavailable(
            "Корпоративная система ПромТеха недоступна (имитация сбоя)"
        )


app = MCPServer(
    name="promtech-equipment",
    instructions=(
        "Учебный MCP-сервер ПромТеха: узнать статус оборудования и "
        "зарегистрировать инцидент без прямого доступа к боевой системе."
    ),
)


@app.resource("equipment://list")
def list_equipment() -> str:
    """Ресурс: полный реестр оборудования для чтения хостом без вызова tool.

    Отличие ресурса от инструмента: инструмент — это действие с аргументами
    («покажи статус вот этого id»), ресурс — просто данные по фиксированному
    адресу (URI), которые хост может прочитать заранее, не дожидаясь вопроса
    пользователя.
    """
    return json.dumps(list(EQUIPMENT.values()), ensure_ascii=False, indent=2)


# ====================================================================
# инструмент чтения: get_equipment_status
# ====================================================================

@app.tool()
def get_equipment_status(equipment_id: str) -> dict:
    result = {}
    try:
        if equipment_id not in EQUIPMENT:
            result = {"ok": False, "error": f"Оборудование {equipment_id} не найдено"}
        else:
            _simulate_corporate_lookup(equipment_id)
            result = {"ok": True, **EQUIPMENT[equipment_id]}
        log_action("get_equipment_status", {"equipment_id": equipment_id}, result)
        return result

    except CorporateSystemUnavailable as exc:
        result = {"ok": False, "error": str(exc)}
        log_action("get_equipment_status", {"equipment_id": equipment_id}, result)
        return result


# ====================================================================
# инструмент записи: create_incident
# ====================================================================

@app.tool()
def create_incident(
    equipment_id: str,
    description: str,
    priority: str,
    role: str = "viewer",
) -> dict:
    result = {}
    if role not in ALLOWED_TO_CREATE_INCIDENT:
        result = {
            "ok": False,
            "error": f"Роль {role!r} не имеет прав на создание инцидентов",
        }
    elif equipment_id not in EQUIPMENT:
        result = {"ok": False, "error": f"Оборудование {equipment_id} не найдено"}
    elif priority not in ALLOWED_PRIORITIES:
        result = {
            "ok": False,
            "error": f"Недопустимый приоритет {priority!r},"
            f" ожидается один из {sorted(ALLOWED_PRIORITIES)}",
        }
    else:
        incident = {
            "id": uuid.uuid4().hex[:8],
            "equipment_id": equipment_id,
            "description": description,
            "priority": priority,
            "status": "open",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_incident(incident)
        result = {"ok": True, "incident": incident}
    log_action(
        "create_incident",
        {"equipment_id": equipment_id, "priority": priority, "role": role},
        result,
    )
    return result

if __name__ == "__main__":
    app.run(transport="stdio")
