"""The 12 supported languages (PLAN.md §6.1) and the paragraph each user reads aloud: about
30 s of their own speech, used as negatives. The text only has to be natural speech in the
language; it must not contain the wake word, so it is plain everyday prose."""

from __future__ import annotations

LANGUAGES: dict[str, dict] = {
    "en": {
        "name": "English", "native_name": "English", "rtl": False, "needs_review": False,
        "paragraph": (
            "The morning was quiet, and the streets were still wet from the rain. A baker opened "
            "the shop on the corner, and the smell of fresh bread filled the air. Children walked "
            "to school with their bags on their backs, talking about the weekend. An old man sat "
            "on a bench, reading the newspaper and watching the pigeons. Slowly, the city woke up, "
            "and a new day began."
        ),
    },
    "de": {
        "name": "German", "native_name": "Deutsch", "rtl": False, "needs_review": False,
        "paragraph": (
            "Der Morgen war ruhig, und die Straßen waren noch nass vom Regen. Ein Bäcker öffnete "
            "seinen Laden an der Ecke, und der Duft von frischem Brot lag in der Luft. Kinder "
            "gingen mit ihren Rucksäcken zur Schule und sprachen über das Wochenende. Ein alter "
            "Mann saß auf einer Bank, las die Zeitung und beobachtete die Tauben. Langsam erwachte "
            "die Stadt, und ein neuer Tag begann."
        ),
    },
    "fr": {
        "name": "French", "native_name": "Français", "rtl": False, "needs_review": False,
        "paragraph": (
            "La matinée était calme, et les rues étaient encore mouillées par la pluie. Un "
            "boulanger a ouvert sa boutique au coin de la rue, et l'odeur du pain frais a rempli "
            "l'air. Les enfants marchaient vers l'école, le sac sur le dos, en parlant du "
            "week-end. Un vieil homme était assis sur un banc, il lisait le journal et regardait "
            "les pigeons. Peu à peu, la ville s'est réveillée, et une nouvelle journée a commencé."
        ),
    },
    "ca": {
        "name": "Catalan", "native_name": "Català", "rtl": False, "needs_review": False,
        "paragraph": (
            "El matí era tranquil, i els carrers encara estaven molls per la pluja. Un forner va "
            "obrir la botiga de la cantonada, i l'olor de pa acabat de fer va omplir l'aire. Els "
            "nens anaven a l'escola amb la motxilla a l'esquena, parlant del cap de setmana. Un "
            "home gran seia en un banc, llegint el diari i mirant els coloms. A poc a poc, la "
            "ciutat es va despertar, i va començar un nou dia."
        ),
    },
    "rw": {
        "name": "Kinyarwanda", "native_name": "Ikinyarwanda", "rtl": False, "needs_review": True,
        "paragraph": (
            "Mu gitondo hari hatuje, kandi imihanda yari ikiri itose kubera imvura. Umutetsi "
            "w'imigati yafunguye iduka rye ku nkengero y'umuhanda, maze impumuro y'imigati mishya "
            "yuzura mu kirere. Abana bajyaga ku ishuri bafite ibikapu ku mugongo, baganira ku "
            "mpera z'icyumweru. Umusaza yari yicaye ku ntebe, asoma ikinyamakuru kandi yitegereza "
            "inuma. Buhoro buhoro, umujyi warakangutse, maze umunsi mushya uratangira."
        ),
    },
    "es": {
        "name": "Spanish", "native_name": "Español", "rtl": False, "needs_review": False,
        "paragraph": (
            "La mañana estaba tranquila, y las calles todavía estaban mojadas por la lluvia. Un "
            "panadero abrió su tienda en la esquina, y el olor a pan recién hecho llenó el aire. "
            "Los niños caminaban hacia la escuela con sus mochilas, hablando del fin de semana. "
            "Un anciano estaba sentado en un banco, leyendo el periódico y mirando a las palomas. "
            "Poco a poco, la ciudad despertó, y comenzó un nuevo día."
        ),
    },
    "fa": {
        "name": "Persian", "native_name": "فارسی", "rtl": True, "needs_review": True,
        "paragraph": (
            "صبح آرام بود و خیابان‌ها هنوز از باران خیس بودند. نانوایی مغازه‌اش را سر کوچه باز "
            "کرد و بوی نان تازه در هوا پیچید. بچه‌ها با کوله‌پشتی به مدرسه می‌رفتند و دربارهٔ "
            "آخر هفته حرف می‌زدند. پیرمردی روی نیمکتی نشسته بود، روزنامه می‌خواند و کبوترها را "
            "تماشا می‌کرد. کم‌کم شهر بیدار شد و روزی تازه آغاز شد."
        ),
    },
    "it": {
        "name": "Italian", "native_name": "Italiano", "rtl": False, "needs_review": False,
        "paragraph": (
            "La mattina era tranquilla, e le strade erano ancora bagnate dalla pioggia. Un "
            "fornaio ha aperto il suo negozio all'angolo, e il profumo del pane fresco ha "
            "riempito l'aria. I bambini andavano a scuola con lo zaino in spalla, parlando del "
            "fine settimana. Un anziano era seduto su una panchina, leggeva il giornale e "
            "guardava i piccioni. Piano piano, la città si è svegliata, ed è cominciata una nuova "
            "giornata."
        ),
    },
    "ru": {
        "name": "Russian", "native_name": "Русский", "rtl": False, "needs_review": False,
        "paragraph": (
            "Утро было тихим, и улицы ещё не высохли после дождя. Пекарь открыл свою лавку на "
            "углу, и в воздухе запахло свежим хлебом. Дети шли в школу с рюкзаками за спиной и "
            "говорили о выходных. Пожилой мужчина сидел на скамейке, читал газету и смотрел на "
            "голубей. Понемногу город просыпался, и начинался новый день."
        ),
    },
    "pl": {
        "name": "Polish", "native_name": "Polski", "rtl": False, "needs_review": False,
        "paragraph": (
            "Poranek był cichy, a ulice wciąż były mokre od deszczu. Piekarz otworzył swój sklep "
            "na rogu i zapach świeżego chleba wypełnił powietrze. Dzieci szły do szkoły z "
            "plecakami, rozmawiając o weekendzie. Starszy pan siedział na ławce, czytał gazetę i "
            "obserwował gołębie. Powoli miasto się budziło i zaczynał się nowy dzień."
        ),
    },
    "eu": {
        "name": "Basque", "native_name": "Euskara", "rtl": False, "needs_review": True,
        "paragraph": (
            "Goiza lasaia zen, eta kaleak oraindik bustita zeuden euriaren ondorioz. Okin batek "
            "izkinako denda ireki zuen, eta ogi berriaren usainak airea bete zuen. Haurrak "
            "eskolara zihoazen motxila bizkarrean, asteburuaz hizketan. Gizon zahar bat bankuan "
            "eserita zegoen, egunkaria irakurtzen eta usoei begira. Poliki-poliki, hiria esnatu "
            "zen, eta egun berri bat hasi zen."
        ),
    },
    "cy": {
        "name": "Welsh", "native_name": "Cymraeg", "rtl": False, "needs_review": True,
        "paragraph": (
            "Roedd y bore'n dawel, ac roedd y strydoedd yn dal yn wlyb ar ôl y glaw. Agorodd "
            "pobydd ei siop ar y gornel, a llenwodd arogl bara ffres yr awyr. Roedd y plant yn "
            "cerdded i'r ysgol gyda'u bagiau ar eu cefnau, yn siarad am y penwythnos. Eisteddai "
            "hen ddyn ar fainc, yn darllen y papur newydd ac yn gwylio'r colomennod. Yn araf, "
            "deffrodd y ddinas, a dechreuodd diwrnod newydd."
        ),
    },
}
