"""Единый словарь военных классов для YOLO-World (88 классов).
ВАЖНО: при изменении списка НУЖНО заново экспортировать ONNX!"""

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