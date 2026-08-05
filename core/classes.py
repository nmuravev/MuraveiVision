"""Единый словарь военных классов для YOLO-World (88 классов).
ВАЖНО: при изменении списка НУЖНО заново экспортировать ONNX!"""

# ── Русские переводы классов (для отображения в GUI/HTML/JSON) ──
# Оригинальный MILITARY_CLASSES НЕ МЕНЯЕТСЯ — он нужен для YOLO.
CLASSES_RU = {
    # ── Бронетехника и авиация ──
    "tank": "танк",
    "tank turret in ground": "вкопанная башня",
    "dug-in artillery": "окопанная артиллерия",
    "armored vehicle": "бронемашина",
    "armored personnel carrier": "БТР",
    "vehicle with cage armor": "техника с решётчатой бронёй",
    "military truck": "военный грузовик",
    "military jeep": "военный джип",
    "tanker truck": "цистерна",
    "self-propelled artillery": "самоходная артиллерия",
    "multiple rocket launcher": "РСЗО",
    "anti-aircraft gun": "зенитное орудие",
    "howitzer": "гаубица",
    "missile launcher": "пусковая установка",
    "military helicopter": "военный вертолёт",
    "military aircraft": "военный самолёт",
    "quadcopter drone": "квадрокоптер",
    "military boat": "военный катер",
    "inflatable decoy": "надувная ложная цель",
    # ── Личный состав и активность ──
    "soldier": "солдат",
    "person in camouflage uniform": "человек в камуфляже",
    "armed person": "вооружённый человек",
    "person carrying rifle": "человек с винтовкой",
    "person with shovel": "человек с лопатой",
    "soldier digging trench": "солдат копает окоп",
    "group of soldiers": "группа солдат",
    # ── Вооружение, снаряжение, хозяйство ──
    "rifle": "винтовка",
    "machine gun": "пулемёт",
    "grenade launcher": "гранатомёт",
    "anti-tank missile": "противотанковая ракета",
    "ammunition box": "ящик с боеприпасами",
    "wooden crate": "деревянный ящик",
    "metal barrel": "металлическая бочка",
    "jerry can": "канистра",
    "portable generator": "переносной генератор",
    "field kitchen": "полевая кухня",
    "wooden pallet": "деревянный поддон",
    # ── Антенны, РЭБ, связь ──
    "military radio antenna": "военная радиоантенна",
    "radio mast": "радиомачта",
    "satellite dish": "спутниковая тарелка",
    "radar": "радар",
    "electronic warfare antenna": "антенна РЭБ",
    "communication antenna": "антенна связи",
    # ── Укрепления и заграждения ──
    "trench": "окоп",
    "bunker": "бункер",
    "dugout entrance": "вход в блиндаж",
    "foxhole": "окопчик",
    "barbed wire": "колючая проволока",
    "barbed wire coil": "моток колючей проволоки",
    "anti-tank obstacle": "противотанковое заграждение",
    "dragon teeth obstacle": "зубы дракона",
    "Czech hedgehog obstacle": "чешский ёж",
    "concrete block barricade": "бетонный блок",
    "gabion barrier": "габион",
    "sandbag wall": "мешки с песком",
    "checkpoint": "блокпост",
    "watchtower": "вышка",
    "earth berm": "земляной вал",
    "anti-tank ditch": "противотанковый ров",
    # ── Маскировка и укрытия ──
    "camouflage net": "маскировочная сеть",
    "camouflaged position": "замаскированная позиция",
    "vehicle under net": "техника под сетью",
    "covered vehicle": "укрытая техника",
    "military tent": "военная палатка",
    "medical tent": "медицинская палатка",
    "command tent": "командная палатка",
    "anti-drone net": "противодронная сеть",
    "drone protection cage": "защитная клетка от дронов",
    "metal grille armor": "металлическая решётка",
    "cut branches on ground": "срезанные ветки на земле",
    # ── СЛЕДЫ ПРИСУТСТВИЯ ──
    "plastic bottle": "пластиковая бутылка",
    "metal can": "металлическая банка",
    "food wrapper": "обёртка от еды",
    "discarded paper": "брошенная бумага",
    "cardboard box": "картонная коробка",
    "plastic bag": "пластиковый пакет",
    "garbage pile": "куча мусора",
    "campfire remains": "остатки костра",
    "smoke plume": "дым",
    "dust cloud": "облако пыли",
    "tire track": "след от шин",
    "footpath in grass": "тропинка в траве",
    "trampled vegetation": "вытоптанная растительность",
    "disturbed soil": "нарушенный грунт",
    "fresh dug dirt": "свежевырытая земля",
    "discarded clothing": "брошенная одежда",
}


def ru(class_en):
    """Возвращает русский перевод класса или оригинал, если перевода нет."""
    return CLASSES_RU.get(class_en, class_en)


MILITARY_CLASSES = [
    # ── Бронетехника и авиация ──
    "tank",
    "tank turret in ground",        # вкопанная башня — классика скрытых позиций
    "dug-in artillery",             # окопанная артиллерия
    "armored vehicle",
    "armored personnel carrier",
    "vehicle with cage armor",      # те самые "решётки"/мангалы на технике
    "military truck",
    "military jeep",
    "tanker truck",
    "self-propelled artillery",
    "multiple rocket launcher",
    "anti-aircraft gun",
    "howitzer",
    "missile launcher",
    "military helicopter",
    "military aircraft",
    "quadcopter drone",
    "military boat",
    "inflatable decoy",             # ложные цели — враг обманывает, мы ищем обман

    # ── Личный состав и активность ──
    "soldier",
    "person in camouflage uniform",
    "armed person",
    "person carrying rifle",
    "person with shovel",           # кто-то копает = свежая позиция
    "soldier digging trench",
    "group of soldiers",

    # ── Вооружение, снаряжение, хозяйство ──
    "rifle",
    "machine gun",
    "grenade launcher",
    "anti-tank missile",
    "ammunition box",
    "wooden crate",
    "metal barrel",
    "jerry can",                    # топливные канистры — вечный спутник позиций
    "portable generator",           # генератор: без него ни РЭБ, ни связь
    "field kitchen",                # полевая кухня = люди рядом
    "wooden pallet",

    # ── Антенны, РЭБ, связь ──
    "military radio antenna",
    "radio mast",
    "satellite dish",
    "radar",
    "electronic warfare antenna",
    "communication antenna",

    # ── Укрепления и заграждения ──
    "trench",
    "bunker",
    "dugout entrance",
    "foxhole",
    "barbed wire",
    "barbed wire coil",
    "anti-tank obstacle",
    "dragon teeth obstacle",
    "Czech hedgehog obstacle",
    "concrete block barricade",
    "gabion barrier",               # габионы/HESCO — визитка современных укрепов
    "sandbag wall",
    "checkpoint",
    "watchtower",
    "earth berm",
    "anti-tank ditch",

    # ── Маскировка и укрытия ──
    "camouflage net",
    "camouflaged position",
    "vehicle under net",
    "covered vehicle",
    "military tent",
    "medical tent",
    "command tent",
    "anti-drone net",               # противодронные сети/экраны
    "drone protection cage",
    "metal grille armor",
    "cut branches on ground",       # срезанные ветки = свежая маскировка

    # ── СЛЕДЫ ПРИСУТСТВИЯ (мусор и демаскирующие признаки) ──
    "plastic bottle",
    "metal can",
    "food wrapper",
    "discarded paper",
    "cardboard box",
    "plastic bag",
    "garbage pile",
    "campfire remains",
    "smoke plume",                  # дым: костёр, генератор, подбитая техника
    "dust cloud",                   # пыль от движения техники по грунту
    "tire track",
    "footpath in grass",            # тропинка там, где её "не должно быть"
    "trampled vegetation",
    "disturbed soil",
    "fresh dug dirt",
    "discarded clothing",
]