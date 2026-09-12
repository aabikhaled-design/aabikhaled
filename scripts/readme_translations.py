# -*- coding: utf-8 -*-
"""Hand-authored README translations, consumed by build_readme_i18n.py.

Keyed by the exact English block (whitespace-normalized). Inline markdown, code
spans, and links are preserved in the translations. The high-visibility landing
copy (hero pitch, section headings, closers) is translated; long technical
paragraphs stay in canonical English. Any block without an entry falls back to
English, so partial coverage is safe and this table can grow language by
language without breaking a build.

To add a language: add its code here and to the README language bar; run
    python3 scripts/build_readme_i18n.py
"""

# --- exact English block keys (must match build_readme_i18n.py normalization) ---
HERO1 = "**84% of students already use AI tools. Only 18% feel prepared to use them professionally.** This curriculum closes that gap."
HERO2 = "523 lessons. 20 phases. ~342 hours. Python, TypeScript, Rust, Julia. Every lesson ships a reusable artifact: a prompt, a skill, an agent, an MCP server. Free, open source, MIT."
HERO3 = "You don't just learn AI. You build it. End-to-end. By hand."
H_START_BUILD = "Start here: choose what you want to build"
START_BUILD = "You do not need to scan 523 lessons before beginning. Pick one goal. Each link opens the same curriculum on GitHub or the website, and both versions use the same lesson code."
NOT_SURE = "Not sure where you fit? Use the [`start-learning` placement tutor](skills/start-learning/SKILL.md) or the [website prerequisites guide](https://aiengineeringfromscratch.com/prereqs.html)."
LEARNING_PATHS = "Compare four core domains and six career routes in the [AI Engineering Learning Paths](https://aiengineeringfromscratch.com/learning-paths.html)."
H_SPONSORS = "Sponsors"
SPONSOR_ALT = "SerpApi. Web Search API for your AI apps. Available in Markdown and JSON for any integration."
SPONSOR_THANKS = "Thank you to our sponsors."
SPONSOR_SUPPORT = "Your support keeps every lesson free and open source."
SEE_SUPPORTERS = "See all supporters"
H_USE_LESSON = "Use every lesson the same way"
LESSON_COMMANDS = "Commands in lesson pages are paths from the repository root unless the lesson explicitly says to change directories. If a lesson offers several languages, run the implementation for the language you are learning."
H_CLONE_EVIDENCE = "Clone it and produce your first evidence"
PREFLIGHT = "The preflight separates requirements needed now from tools needed later. Every required failure includes the detected reason and a corrective command. The second command is a dependency-free lesson and ends by showing that a matrix times a vector is the operation inside a neural network layer. Save that terminal output as your first evidence."
WAYS = "Three ways in. Pick one."
LICENSE_LINE = "MIT. Use it however you want — fork it, teach it, sell it, ship it. Attribution appreciated, not required."
MAINTAINED = "Maintained by [Rohit Ghumare](https://github.com/rohitg00) and the community."

H_HOW = "How this works"
H_CURR = "The shape of the curriculum"
H_LESSON = "The shape of a lesson"
H_START = "Getting started"
H_PREREQ = "Prerequisites"
H_BOOK = "Read it as a book"
H_SHIPS = "Every lesson ships something"
H_CONTENTS = "Contents"
H_TOOLKIT = "The toolkit"
H_WHERE = "Where to start"
H_WHY = "Why this matters now"
H_CONTRIB = "Contributing"
H_SPONSOR = "Sponsor the work"
H_STAR = "Star history"
H_LICENSE = "License"
SPONSOR_CLOSING = "Free, MIT-licensed, 523 lessons. Thank you to the sponsors and backers who make the work possible. [See all sponsors and backers](BACKERS.md)."
SPONSOR_INVITE = "Want to support the work? See [sponsorship options](SPONSORS.md), including [hardware sponsorships](SPONSORS.md#hardware-lab-partner), or [sponsor on GitHub](https://github.com/sponsors/rohitg00)."

README_NOTE = {
    "es": '<p align="center"><sub>Traducción de la comunidad. El <a href="../../README.md">inglés es la versión canónica</a> · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "fr": '<p align="center"><sub>Traduction communautaire. L\'<a href="../../README.md">anglais fait foi</a> · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "pt": '<p align="center"><sub>Tradução da comunidade. O <a href="../../README.md">inglês é a versão canônica</a> · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "de": '<p align="center"><sub>Community-Übersetzung. Maßgeblich ist das <a href="../../README.md">englische Original</a> · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "it": '<p align="center"><sub>Traduzione della community. Fa fede la <a href="../../README.md">versione inglese</a> · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "zh": '<p align="center"><sub>社区翻译，以<a href="../../README.md">英文原文为准</a> · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "ja": '<p align="center"><sub>コミュニティによる翻訳です。正文は<a href="../../README.md">英語版</a>です · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "ko": '<p align="center"><sub>커뮤니티 번역입니다. 정본은 <a href="../../README.md">영어판</a>입니다 · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "hi": '<p align="center"><sub>सामुदायिक अनुवाद। <a href="../../README.md">अंग्रेज़ी संस्करण ही प्रामाणिक है</a> · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "ar": '<p align="center" dir="rtl"><sub>ترجمة مجتمعية. النسخة <a href="../../README.md">الإنجليزية هي المرجعية</a> · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "ru": '<p align="center"><sub>Перевод сообщества. Каноничной является <a href="../../README.md">английская версия</a> · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
    "tr": '<p align="center"><sub>Topluluk çevirisi. Esas alınan sürüm <a href="../../README.md">İngilizce</a>dir · <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a></sub></p>',
}

TRANSLATIONS = {
    "es": {
        HERO1: "**El 84 % de los estudiantes ya usa herramientas de IA. Solo el 18 % se siente preparado para usarlas de forma profesional.** Este plan de estudios cierra esa brecha.",
        HERO2: "523 lecciones. 20 fases. ~342 horas. Python, TypeScript, Rust, Julia. Cada lección entrega un artefacto reutilizable: un prompt, una skill, un agente, un servidor MCP. Gratis, código abierto, MIT.",
        HERO3: "No solo aprendes IA. La construyes. De principio a fin. A mano.",
        WAYS: "Tres formas de empezar. Elige una.",
        LICENSE_LINE: "MIT. Úsalo como quieras: bifúrcalo, enséñalo, véndelo, publícalo. La atribución se agradece, pero no es obligatoria.",
        MAINTAINED: "Mantenido por [Rohit Ghumare](https://github.com/rohitg00) y la comunidad.",
        H_HOW: "Cómo funciona", H_CURR: "La forma del plan de estudios", H_LESSON: "La forma de una lección",
        H_START: "Primeros pasos", H_PREREQ: "Requisitos previos", H_BOOK: "Léelo como un libro",
        H_SHIPS: "Cada lección entrega algo", H_CONTENTS: "Contenido", H_TOOLKIT: "El kit de herramientas",
        H_WHERE: "Por dónde empezar", H_WHY: "Por qué esto importa ahora", H_CONTRIB: "Cómo contribuir",
        H_SPONSOR: "Patrocina el proyecto", H_STAR: "Historial de estrellas", H_LICENSE: "Licencia",
    },
    "fr": {
        HERO1: "**84 % des étudiants utilisent déjà des outils d'IA. Seuls 18 % se sentent prêts à les utiliser de façon professionnelle.** Ce cursus comble cet écart.",
        HERO2: "523 leçons. 20 phases. ~342 heures. Python, TypeScript, Rust, Julia. Chaque leçon livre un artefact réutilisable : un prompt, une skill, un agent, un serveur MCP. Gratuit, open source, MIT.",
        HERO3: "Vous n'apprenez pas seulement l'IA. Vous la construisez. De bout en bout. À la main.",
        WAYS: "Trois façons de commencer. Choisissez-en une.",
        LICENSE_LINE: "MIT. Utilisez-le comme vous voulez : forkez-le, enseignez-le, vendez-le, publiez-le. L'attribution est appréciée, mais pas obligatoire.",
        MAINTAINED: "Maintenu par [Rohit Ghumare](https://github.com/rohitg00) et la communauté.",
        H_HOW: "Comment ça marche", H_CURR: "La forme du cursus", H_LESSON: "La forme d'une leçon",
        H_START: "Pour commencer", H_PREREQ: "Prérequis", H_BOOK: "Lisez-le comme un livre",
        H_SHIPS: "Chaque leçon produit quelque chose", H_CONTENTS: "Sommaire", H_TOOLKIT: "La boîte à outils",
        H_WHERE: "Par où commencer", H_WHY: "Pourquoi c'est important maintenant", H_CONTRIB: "Contribuer",
        H_SPONSOR: "Soutenir le projet", H_STAR: "Historique des étoiles", H_LICENSE: "Licence",
    },
    "pt": {
        HERO1: "**84% dos estudantes já usam ferramentas de IA. Apenas 18% se sentem preparados para usá-las profissionalmente.** Este currículo fecha essa lacuna.",
        HERO2: "523 lições. 20 fases. ~342 horas. Python, TypeScript, Rust, Julia. Cada lição entrega um artefato reutilizável: um prompt, uma skill, um agente, um servidor MCP. Grátis, código aberto, MIT.",
        HERO3: "Você não apenas aprende IA. Você a constrói. Do início ao fim. À mão.",
        H_START_BUILD: "Comece aqui: escolha o que você quer construir",
        START_BUILD: "Você não precisa percorrer 523 lições antes de começar. Escolha um objetivo. Cada link abre o mesmo currículo no GitHub ou no site, e as duas versões usam o mesmo código das lições.",
        "| Your goal | Learn on GitHub | Learn on the website |": "| Seu objetivo | Aprenda no GitHub | Aprenda no site |",
        "| I am new and want the complete foundation | [Phase 0: Setup and Tooling](phases/00-setup-and-tooling/) | [Dev Environment](https://aiengineeringfromscratch.com/lesson?path=phases/00-setup-and-tooling/01-dev-environment) |": "| Estou começando e quero a base completa | [Fase 0: Configuração e ferramentas](phases/00-setup-and-tooling/) | [Ambiente de desenvolvimento](https://aiengineeringfromscratch.com/lesson?path=phases/00-setup-and-tooling/01-dev-environment) |",
        "| I know Python and want math plus ML foundations | [Phase 1: Math Foundations](phases/01-math-foundations/) | [Linear Algebra Intuition](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/01-linear-algebra-intuition) |": "| Sei Python e quero fundamentos de matemática e ML | [Fase 1: Fundamentos matemáticos](phases/01-math-foundations/) | [Intuição de álgebra linear](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/01-linear-algebra-intuition) |",
        "| I want to build production LLM applications | [Phase 11: LLM Engineering](phases/11-llm-engineering/) | [Prompt Engineering](https://aiengineeringfromscratch.com/lesson?path=phases/11-llm-engineering/01-prompt-engineering) |": "| Quero criar aplicações de LLM para produção | [Fase 11: Engenharia de LLM](phases/11-llm-engineering/) | [Engenharia de prompts](https://aiengineeringfromscratch.com/lesson?path=phases/11-llm-engineering/01-prompt-engineering) |",
        "| I want to build agents | [Phase 14: Agent Engineering](phases/14-agent-engineering/) | [The Agent Loop](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/01-the-agent-loop) |": "| Quero criar agentes | [Fase 14: Engenharia de agentes](phases/14-agent-engineering/) | [O loop do agente](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/01-the-agent-loop) |",
        "| I want to use coding agents on real repositories | [Agent-Assisted Engineering path](learning-paths/using-coding-agents.json) | [Agent-Assisted Engineering](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/31-agent-workbench-why-models-fail&learningPath=using-coding-agents) |": "| Quero usar agentes de código em repositórios reais | [Trilha de engenharia assistida por agentes](learning-paths/using-coding-agents.json) | [Engenharia assistida por agentes](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/31-agent-workbench-why-models-fail&learningPath=using-coding-agents) |",
        "| I want to shape the right build before implementation | [Product Judgment and Delivery path](learning-paths/shaping-the-build.json) | [Product Judgment and Delivery](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/47-outcomes-before-output&learningPath=shaping-the-build) |": "| Quero definir a solução certa antes de implementar | [Trilha de julgamento de produto e entrega](learning-paths/shaping-the-build.json) | [Julgamento de produto e entrega](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/47-outcomes-before-output&learningPath=shaping-the-build) |",
        "| I want to build with Model Context Protocol (MCP) | [Model Context Protocol (MCP) route](phases/13-tools-and-protocols/README.md#model-context-protocol-mcp-path) | [Model Context Protocol (MCP) path](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/06-mcp-fundamentals&learningPath=model-context-protocol) |": "| Quero desenvolver com o Model Context Protocol (MCP) | [Rota do Model Context Protocol (MCP)](phases/13-tools-and-protocols/README.md#model-context-protocol-mcp-path) | [Trilha do Model Context Protocol (MCP)](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/06-mcp-fundamentals&learningPath=model-context-protocol) |",
        "| I want to write and ship Agent Skills | [Focused Agent Skills route](phases/13-tools-and-protocols/README.md#agent-skills-fast-path) | [Agent Skills path](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/22-skills-and-agent-sdks&learningPath=agent-skills) |": "| Quero escrever e publicar Agent Skills | [Rota focada de Agent Skills](phases/13-tools-and-protocols/README.md#agent-skills-fast-path) | [Trilha de Agent Skills](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/22-skills-and-agent-sdks&learningPath=agent-skills) |",
        "| I want to prepare for a Claude certification | [Certification onboarding](certifications/claude/GETTING_STARTED.md) | [Certification Academy](https://aiengineeringfromscratch.com/certifications.html) |": "| Quero me preparar para uma certificação Claude | [Introdução à certificação](certifications/claude/GETTING_STARTED.md) | [Academia de certificação](https://aiengineeringfromscratch.com/certifications.html) |",
        NOT_SURE: "Não sabe onde se encaixa? Use o [tutor de nivelamento `start-learning`](skills/start-learning/SKILL.md) ou o [guia de pré-requisitos do site](https://aiengineeringfromscratch.com/prereqs.html).",
        LEARNING_PATHS: "Compare quatro domínios centrais e seis rotas de carreira nas [Trilhas de aprendizagem em Engenharia de IA](https://aiengineeringfromscratch.com/learning-paths.html).",
        H_USE_LESSON: "Use todas as lições da mesma maneira",
        "1. **Read** `docs/en.md` and explain the core idea in your own words.": "1. **Leia** `docs/en.md` e explique a ideia central com suas próprias palavras.",
        "2. **Type and build** the important code instead of treating the code block as decoration.": "2. **Digite e construa** o código importante em vez de tratar o bloco de código como decoração.",
        "3. **Run** the lesson command from the repository root, the directory containing `README.md` and `phases/`.": "3. **Execute** o comando da lição na raiz do repositório, o diretório que contém `README.md` e `phases/`.",
        "4. **Keep evidence**: the command, working directory, exit code, meaningful output, and the artifact you changed or produced.": "4. **Guarde evidências**: o comando, o diretório de trabalho, o código de saída, a saída relevante e o artefato alterado ou produzido.",
        "5. **Continue** only when you can explain the output and make one small change without guessing.": "5. **Continue** somente quando conseguir explicar a saída e fazer uma pequena mudança sem adivinhar.",
        LESSON_COMMANDS: "Os comandos nas páginas das lições usam caminhos a partir da raiz do repositório, salvo quando a lição manda mudar de diretório. Se houver várias linguagens, execute a implementação da linguagem que você está aprendendo.",
        H_CLONE_EVIDENCE: "Clone o repositório e produza sua primeira evidência",
        PREFLIGHT: "A verificação inicial separa os requisitos necessários agora das ferramentas necessárias depois. Cada falha obrigatória mostra a causa detectada e um comando corretivo. O segundo comando executa uma lição sem dependências e termina mostrando que multiplicar uma matriz por um vetor é a operação dentro de uma camada de rede neural. Guarde essa saída do terminal como sua primeira evidência.",
        WAYS: "Três formas de começar. Escolha uma.",
        LICENSE_LINE: "MIT. Use como quiser: faça fork, ensine, venda, publique. A atribuição é bem-vinda, mas não obrigatória.",
        MAINTAINED: "Mantido por [Rohit Ghumare](https://github.com/rohitg00) e pela comunidade.",
        H_HOW: "Como funciona", H_CURR: "O formato do currículo", H_LESSON: "O formato de uma lição",
        H_START: "Primeiros passos", H_PREREQ: "Pré-requisitos", H_BOOK: "Leia como um livro",
        H_SHIPS: "Cada lição entrega algo", H_CONTENTS: "Conteúdo", H_TOOLKIT: "O kit de ferramentas",
        H_WHERE: "Por onde começar", H_WHY: "Por que isso importa agora", H_CONTRIB: "Como contribuir",
        H_SPONSOR: "Patrocine o trabalho", H_STAR: "Histórico de estrelas", H_LICENSE: "Licença",
    },
    "de": {
        HERO1: "**84 % der Studierenden nutzen bereits KI-Tools. Nur 18 % fühlen sich bereit, sie professionell einzusetzen.** Dieser Lehrplan schließt diese Lücke.",
        HERO2: "523 Lektionen. 20 Phasen. ~342 Stunden. Python, TypeScript, Rust, Julia. Jede Lektion liefert ein wiederverwendbares Artefakt: einen Prompt, einen Skill, einen Agenten, einen MCP-Server. Kostenlos, Open Source, MIT.",
        HERO3: "Du lernst KI nicht nur. Du baust sie. Von Anfang bis Ende. Von Hand.",
        WAYS: "Drei Einstiege. Wähle einen.",
        LICENSE_LINE: "MIT. Nutze es, wie du willst: forke es, unterrichte es, verkaufe es, veröffentliche es. Nennung ist willkommen, aber nicht erforderlich.",
        MAINTAINED: "Betreut von [Rohit Ghumare](https://github.com/rohitg00) und der Community.",
        H_HOW: "So funktioniert es", H_CURR: "Der Aufbau des Lehrplans", H_LESSON: "Der Aufbau einer Lektion",
        H_START: "Erste Schritte", H_PREREQ: "Voraussetzungen", H_BOOK: "Als Buch lesen",
        H_SHIPS: "Jede Lektion liefert etwas", H_CONTENTS: "Inhalt", H_TOOLKIT: "Das Toolkit",
        H_WHERE: "Wo anfangen", H_WHY: "Warum das jetzt zählt", H_CONTRIB: "Mitwirken",
        H_SPONSOR: "Die Arbeit unterstützen", H_STAR: "Sternverlauf", H_LICENSE: "Lizenz",
    },
    "it": {
        HERO1: "**L'84% degli studenti usa già strumenti di IA. Solo il 18% si sente pronto a usarli professionalmente.** Questo percorso colma quel divario.",
        HERO2: "523 lezioni. 20 fasi. ~342 ore. Python, TypeScript, Rust, Julia. Ogni lezione produce un artefatto riutilizzabile: un prompt, una skill, un agente, un server MCP. Gratis, open source, MIT.",
        HERO3: "Non impari solo l'IA. La costruisci. Dall'inizio alla fine. A mano.",
        WAYS: "Tre modi per iniziare. Scegline uno.",
        LICENSE_LINE: "MIT. Usalo come vuoi: forkalo, insegnalo, vendilo, pubblicalo. L'attribuzione è gradita, ma non obbligatoria.",
        MAINTAINED: "Mantenuto da [Rohit Ghumare](https://github.com/rohitg00) e dalla community.",
        H_HOW: "Come funziona", H_CURR: "La struttura del percorso", H_LESSON: "La struttura di una lezione",
        H_START: "Per iniziare", H_PREREQ: "Prerequisiti", H_BOOK: "Leggilo come un libro",
        H_SHIPS: "Ogni lezione produce qualcosa", H_CONTENTS: "Indice", H_TOOLKIT: "Il toolkit",
        H_WHERE: "Da dove iniziare", H_WHY: "Perché conta adesso", H_CONTRIB: "Contribuire",
        H_SPONSOR: "Sostieni il progetto", H_STAR: "Cronologia delle stelle", H_LICENSE: "Licenza",
    },
    "zh": {
        HERO1: "**84% 的学生已经在使用 AI 工具，却只有 18% 觉得自己能专业地使用它们。** 这套课程正是为了填补这道鸿沟。",
        HERO2: "523 节课。20 个阶段。约 342 小时。Python、TypeScript、Rust、Julia。每节课都产出一个可复用的成果：一个提示词、一个技能、一个智能体、一个 MCP 服务器。免费、开源、MIT 许可。",
        HERO3: "你不只是学 AI，你亲手把它造出来。从头到尾，全部手写。",
        WAYS: "三种入门方式，任选其一。",
        LICENSE_LINE: "MIT 许可。随你怎么用：复刻、教学、出售、发布都行。欢迎署名，但并非必须。",
        MAINTAINED: "由 [Rohit Ghumare](https://github.com/rohitg00) 和社区共同维护。",
        H_HOW: "运作方式", H_CURR: "课程的整体结构", H_LESSON: "单节课的结构",
        H_START: "快速开始", H_PREREQ: "先决条件", H_BOOK: "当作一本书来读",
        H_SHIPS: "每节课都有产出", H_CONTENTS: "目录", H_TOOLKIT: "工具箱",
        H_WHERE: "从哪里开始", H_WHY: "为什么这在当下很重要", H_CONTRIB: "参与贡献",
        H_SPONSOR: "赞助本项目", H_STAR: "Star 历史", H_LICENSE: "许可证",
    },
    "ja": {
        HERO1: "**学生の84%はすでにAIツールを使っていますが、それを専門的に使いこなせると感じているのはわずか18%です。** このカリキュラムはそのギャップを埋めます。",
        HERO2: "523のレッスン。20のフェーズ。約342時間。Python、TypeScript、Rust、Julia。各レッスンは再利用できる成果物を残します。プロンプト、スキル、エージェント、MCPサーバー。無料、オープンソース、MIT。",
        HERO3: "AIをただ学ぶのではありません。自分の手で作ります。最初から最後まで、手作業で。",
        WAYS: "入り方は3つ。ひとつ選んでください。",
        LICENSE_LINE: "MIT。好きなように使ってください。フォークする、教える、売る、公開する。クレジットは歓迎しますが、必須ではありません。",
        MAINTAINED: "[Rohit Ghumare](https://github.com/rohitg00) とコミュニティが保守しています。",
        H_HOW: "仕組み", H_CURR: "カリキュラムの全体像", H_LESSON: "レッスンの構成",
        H_START: "はじめに", H_PREREQ: "前提知識", H_BOOK: "本として読む",
        H_SHIPS: "どのレッスンにも成果物がある", H_CONTENTS: "目次", H_TOOLKIT: "ツールキット",
        H_WHERE: "どこから始めるか", H_WHY: "なぜ今これが重要なのか", H_CONTRIB: "コントリビュート",
        H_SPONSOR: "プロジェクトを支援する", H_STAR: "スター履歴", H_LICENSE: "ライセンス",
    },
    "ko": {
        HERO1: "**학생의 84%는 이미 AI 도구를 사용하지만, 이를 전문적으로 다룰 준비가 되었다고 느끼는 사람은 18%뿐입니다.** 이 커리큘럼이 그 간극을 메웁니다.",
        HERO2: "523개 레슨. 20개 단계. 약 342시간. Python, TypeScript, Rust, Julia. 모든 레슨은 재사용 가능한 결과물을 남깁니다. 프롬프트, 스킬, 에이전트, MCP 서버. 무료, 오픈소스, MIT.",
        HERO3: "AI를 배우기만 하는 것이 아닙니다. 직접 만듭니다. 처음부터 끝까지, 손으로.",
        WAYS: "시작하는 방법은 세 가지. 하나를 고르세요.",
        LICENSE_LINE: "MIT. 원하는 대로 쓰세요. 포크하고, 가르치고, 팔고, 배포하세요. 출처 표기는 환영하지만 필수는 아닙니다.",
        MAINTAINED: "[Rohit Ghumare](https://github.com/rohitg00) 와 커뮤니티가 관리합니다.",
        H_HOW: "동작 방식", H_CURR: "커리큘럼의 구조", H_LESSON: "레슨의 구조",
        H_START: "시작하기", H_PREREQ: "사전 요구 사항", H_BOOK: "책으로 읽기",
        H_SHIPS: "모든 레슨은 결과물을 남깁니다", H_CONTENTS: "목차", H_TOOLKIT: "툴킷",
        H_WHERE: "어디서 시작할까", H_WHY: "왜 지금 중요한가", H_CONTRIB: "기여하기",
        H_SPONSOR: "프로젝트 후원하기", H_STAR: "스타 히스토리", H_LICENSE: "라이선스",
        # Full coverage of the remaining translatable blocks. Korean here follows
        # the fluent-korean writing rules vendored in .claude/output-styles/:
        # complete sentences with their endings, no dropped particles, no em
        # dashes, and technical terms left in their established English form.
        'From the creator of [Agent Memory - #1 Persistent memory ⭐](https://github.com/rohitg00/agentmemory) <a href="https://github.com/rohitg00/agentmemory/stargazers"><img src="https://img.shields.io/github/stars/rohitg00/agentmemory?style=flat-square&labelColor=fafaf5&color=3553ff" alt="GitHub stars"></a> which naturally works with any agents or chat assistants.':
            '[Agent Memory - 지속 메모리 1위 ⭐](https://github.com/rohitg00/agentmemory) <a href="https://github.com/rohitg00/agentmemory/stargazers"><img src="https://img.shields.io/github/stars/rohitg00/agentmemory?style=flat-square&labelColor=fafaf5&color=3553ff" alt="GitHub stars"></a> 를 만든 사람이 이 커리큘럼도 만들었습니다. Agent Memory는 어떤 에이전트나 챗 어시스턴트와도 그대로 연동됩니다.',
        H_START_BUILD: '여기에서 시작하십시오. 무엇을 만들지 먼저 고르십시오',
        START_BUILD: '시작하기 전에 523개 레슨을 전부 훑어볼 필요는 없습니다. 목표를 하나만 고르십시오. 아래의 각 링크는 GitHub이나 웹사이트에서 동일한 커리큘럼을 열어 주며, 두 판본은 같은 레슨 코드를 사용합니다.',
        NOT_SURE: '자신이 어디에서 시작해야 할지 판단하기 어렵다면, [`start-learning` 배치 진단 튜터](skills/start-learning/SKILL.md)를 사용하거나 [웹사이트의 선행 조건 안내](https://aiengineeringfromscratch.com/prereqs.html)를 참고하십시오.',
        LEARNING_PATHS: '네 개의 핵심 분야와 여섯 개의 커리어 경로를 [AI Engineering 학습 경로](https://aiengineeringfromscratch.com/learning-paths.html)에서 비교해 보십시오.',
        H_USE_LESSON: '모든 레슨은 같은 방식으로 사용합니다',
        LESSON_COMMANDS: '레슨 문서에 나오는 명령은, 레슨이 디렉터리를 옮기라고 명시하지 않는 한 저장소 최상위 디렉터리를 기준으로 삼은 경로입니다. 한 레슨이 여러 언어로 구현을 제공한다면, 여러분이 학습하고 있는 언어의 구현을 실행하십시오.',
        H_CLONE_EVIDENCE: '저장소를 클론해서 첫 번째 결과를 직접 만들어 보십시오',
        PREFLIGHT: '사전 점검 스크립트는 지금 당장 필요한 요구 사항과 나중에 필요한 도구를 구분해서 알려 줍니다. 필수 항목이 하나라도 실패하면, 실패를 일으킨 원인과 그것을 바로잡는 명령을 함께 출력합니다. 두 번째 명령은 외부 의존성이 전혀 없는 레슨을 실행하며, 행렬과 벡터의 곱이 곧 신경망 레이어 내부에서 일어나는 연산이라는 사실을 보여 주면서 끝납니다. 그 터미널 출력을 첫 번째 학습 결과로 저장해 두십시오.',
        'Add the AI tutor in 30 seconds':
            'AI 튜터를 30초 만에 추가하기',
        'If Node.js, `npx`, and a skill-capable coding agent are already installed, your coding agent can become your tutor in two commands. A repository clone is not needed to install or read the tutor. Runnable focused-path labs need `python3`. Agent Skills host labs also need a selected host and a writable user or project skill scope.':
            'Node.js와 `npx`, 그리고 스킬을 지원하는 코딩 에이전트가 이미 설치되어 있다면, 명령 두 줄만으로 그 코딩 에이전트를 학습 튜터로 바꿀 수 있습니다. 튜터를 설치하거나 읽는 데에는 저장소를 클론할 필요가 없습니다. 실행 가능한 집중 경로 실습에는 `python3`이 필요합니다. Agent Skills 호스트 실습에는 호스트를 하나 선택해야 하고, 쓰기가 가능한 사용자 스킬 범위나 프로젝트 스킬 범위도 있어야 합니다.',
        'Check the local requirements first:':
            '먼저 로컬 환경이 요구 사항을 충족하는지 확인하십시오.',
        'Then install the curriculum skills and choose the host and scope you intend to use when the installer asks:':
            '그다음 커리큘럼 스킬을 설치하고, 설치 프로그램이 물어볼 때 사용하려는 호스트와 범위를 선택하십시오.',
        'Invocation syntax belongs to the host, not to the portable `SKILL.md` format:':
            '스킬을 호출하는 문법은 이식 가능한 `SKILL.md` 형식이 아니라 각 호스트가 정합니다.',
        'A ten-question placement quiz maps what you already know to a starting phase and saves a personalized study plan to `LEARNING.md`. From there, the `learn` skill teaches one lesson per session: concept, math, code, quiz. It streams lessons straight from this repo, and the `course-guide` skill jumps you to the exact lesson that covers anything you are stuck on. In Codex, invoke these skills with `learn` and `course-guide`; in Claude Code, use `/learn` and `/course-guide`; in other compatible hosts, ask to use the skill by name.':
            '열 문항짜리 배치 진단 퀴즈가 여러분이 이미 알고 있는 내용을 파악해서 시작할 페이즈를 정해 주고, 개인별 학습 계획을 `LEARNING.md`에 저장합니다. 그다음부터는 `learn` 스킬이 한 세션에 한 레슨씩 가르칩니다. 개념을 설명하고, 수학을 유도하고, 코드를 작성하고, 퀴즈를 냅니다. 이 스킬은 레슨을 이 저장소에서 바로 가져오며, 막히는 내용이 생기면 `course-guide` 스킬이 그 내용을 다루는 레슨으로 곧바로 데려다줍니다. Codex에서는 `learn`과 `course-guide`라고 입력해서 호출하고, Claude Code에서는 `/learn`과 `/course-guide`를 사용하며, 다른 호환 호스트에서는 스킬 이름을 말해서 사용을 요청하십시오.',
        'Only want Model Context Protocol (MCP)? Use the MCP invocation for your host. It creates `MCP-LEARNING.md` and follows one 17-lesson route through stateless requests, transports, bidirectional work, security, reliability, registry governance, and conformance evidence. The exact order and checkpoints live in the [Model Context Protocol (MCP) manifest](learning-paths/model-context-protocol.json).':
            'Model Context Protocol(MCP)만 배우고 싶다면, 사용하는 호스트에 맞는 MCP 호출 방법을 쓰십시오. 그러면 `MCP-LEARNING.md` 파일이 만들어지고, 17개 레슨으로 이루어진 하나의 경로를 따라가게 됩니다. 이 경로는 상태를 유지하지 않는 요청에서 출발해 전송 계층, 양방향 통신, 보안, 신뢰성, 레지스트리 거버넌스, 적합성 증거까지 이어집니다. 정확한 순서와 점검 지점은 [Model Context Protocol(MCP) 매니페스트](learning-paths/model-context-protocol.json)에 적혀 있습니다.',
        'Only want Agent Skills? Use the Agent Skills invocation for your host. It creates `AGENT-SKILLS-LEARNING.md` and follows one coherent five-lesson route: contract, discovery, invocation, sandbox boundaries, then release evals and real-host portability. Start on the web with the [Agent Skills path](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/22-skills-and-agent-sdks&learningPath=agent-skills).':
            'Agent Skills만 배우고 싶다면, 사용하는 호스트에 맞는 Agent Skills 호출 방법을 쓰십시오. 그러면 `AGENT-SKILLS-LEARNING.md` 파일이 만들어지고, 서로 이어지는 다섯 개 레슨을 차례로 학습하게 됩니다. 스킬 패키지의 계약을 정의하고, 스킬을 발견하게 만들고, 실제로 호출하고, 샌드박스 경계를 정하고, 마지막으로 배포 전 평가와 실제 호스트 사이의 이식성을 다룹니다. 웹에서는 [Agent Skills 경로](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/22-skills-and-agent-sdks&learningPath=agent-skills)에서 시작할 수 있습니다.',
        'The installer lists the hosts it can configure and asks where to install. If you do not have Node.js, `npx`, `python3`, a supported host, or a writable scope yet, use the website or read `docs/en.md` manually. That path teaches the concepts, but real-host discovery, invocation, script, and uninstall evidence remains pending until the preflight is available. Read the lessons at [aiengineeringfromscratch.com](https://aiengineeringfromscratch.com).':
            '설치 프로그램은 설정할 수 있는 호스트 목록을 보여 주고 어디에 설치할지 묻습니다. Node.js나 `npx`, `python3`, 지원되는 호스트, 쓰기 가능한 범위 가운데 아직 갖추지 못한 것이 있다면, 웹사이트를 이용하거나 `docs/en.md`를 직접 읽으십시오. 그렇게 해도 개념은 배울 수 있지만, 실제 호스트에서 스킬이 발견되고 호출되고 스크립트가 실행되고 제거되는 과정을 증거로 남기는 일은 사전 점검을 통과한 뒤로 미뤄집니다. 레슨은 [aiengineeringfromscratch.com](https://aiengineeringfromscratch.com)에서 읽을 수 있습니다.',
        "Most AI material teaches in scattered pieces. A paper here, a fine-tuning post there, a flashy agent demo somewhere else. The pieces rarely line up. You ship a chatbot but can't explain its loss curve. You hook a function to an agent but can't say what attention does inside the model that's calling it.":
            'AI를 다루는 자료는 대부분 흩어진 조각으로 가르칩니다. 여기에는 논문이 하나 있고, 저기에는 파인튜닝을 설명하는 글이 하나 있으며, 또 다른 곳에는 화려한 에이전트 데모가 하나 있습니다. 그 조각들은 서로 좀처럼 맞물리지 않습니다. 그래서 챗봇을 출시하고도 그 손실 곡선이 왜 그렇게 움직이는지 설명하지 못합니다. 에이전트에 함수를 연결해 놓고도, 그 함수를 호출하는 모델 내부에서 어텐션이 무슨 일을 하는지 말하지 못합니다.',
        "This curriculum is the spine. 20 phases, 523 lessons, four languages: Python, TypeScript, Rust, Julia. Linear algebra at one end, autonomous swarms at the other. Every algorithm gets built from raw math first. Backprop. Tokenizer. Attention. Agent loop. By the time PyTorch shows up, you already know what it's doing under the hood.":
            '이 커리큘럼은 그 조각들을 하나로 잇는 중심 구조입니다. 20개 페이즈, 523개 레슨, 그리고 Python, TypeScript, Rust, Julia까지 네 가지 언어를 사용합니다. 한쪽 끝에는 선형대수가 있고, 반대쪽 끝에는 자율 군집이 있습니다. 모든 알고리즘은 날것의 수학에서부터 먼저 구현합니다. 역전파를 구현하고, 토크나이저를 구현하고, 어텐션을 구현하고, 에이전트 루프를 구현합니다. PyTorch가 등장할 무렵이면, 여러분은 이미 PyTorch가 내부에서 무엇을 수행하는지 알고 있게 됩니다.',
        'Each lesson runs the same loop: read the problem, derive the math, write the code, run the test, keep the artifact. No five-minute videos, no copy-paste deploys, no hand-holding. Free, open source, and built to run on your own laptop.':
            '모든 레슨은 동일한 절차를 따릅니다. 문제를 읽고, 수학을 유도하고, 코드를 작성하고, 테스트를 실행하고, 그 결과물을 보관합니다. 5분짜리 영상도 없고, 복사해서 붙여넣는 배포도 없으며, 하나하나 떠먹여 주는 안내도 없습니다. 이 커리큘럼은 무료이고 오픈 소스이며, 여러분의 노트북에서 그대로 실행되도록 만들었습니다.',
        "Twenty phases stack on top of each other. Math is the floor. Agents and production are the roof. Skip ahead if you already know the lower layers, but don't skip and then wonder why something at the top is breaking.":
            '20개 페이즈는 아래에서 위로 차곡차곡 쌓여 있습니다. 가장 아래층은 수학이고, 가장 위층은 에이전트와 프로덕션 운영입니다. 아래층을 이미 알고 있다면 건너뛰어도 됩니다. 다만 건너뛴 다음에 위층에서 무언가 깨졌을 때 그 원인을 의아해하는 일은 없어야 합니다.',
        'Each lesson lives in its own folder, with the same structure across the entire curriculum:':
            '모든 레슨은 각자의 폴더에 들어 있으며, 커리큘럼 전체가 동일한 구조를 따릅니다.',
        'Every lesson follows six beats. The *Build It / Use It* split is the spine — you implement the algorithm from scratch first, then run the same thing through the production library. You understand what the framework is doing because you wrote the smaller version yourself.':
            '모든 레슨은 여섯 단계로 진행됩니다. 그중 *Build It / Use It* 구분이 핵심입니다. 먼저 알고리즘을 맨바닥에서 직접 구현하고, 그다음에 똑같은 동작을 실제 프로덕션 라이브러리로 실행해 봅니다. 작은 판본을 직접 작성해 봤기 때문에, 프레임워크가 무슨 일을 하는지 이해하게 됩니다.',
        '**Option A — learn in your terminal *(recommended)*.** After the Node.js, `npx`, host, and scope preflight above, install the learning skills into a compatible agent and let the course drive itself:':
            '**선택지 A: 터미널에서 학습하기 *(권장)*.** 위에서 설명한 Node.js와 `npx`, 호스트, 설치 범위 사전 점검을 마쳤다면, 학습 스킬을 호환되는 에이전트에 설치하고 커리큘럼이 스스로 진행되도록 맡기십시오.',
        'Use the host-specific invocation table above. The installed skills provide `start-learning`, `learn`, `course-guide`, and the focused `learn-mcp` and `learn-agent-skills` routes. Lesson prose can stream from this repository without a clone. A local clone is required for copied repository code commands and executable MCP or Agent Skills labs. Progress lives in `LEARNING.md`, `MCP-LEARNING.md`, or `AGENT-SKILLS-LEARNING.md` in your project, so every session can resume.':
            '위에 있는 호스트별 호출 방법 표를 참고하십시오. 설치된 스킬은 `start-learning`, `learn`, `course-guide`와 함께 집중 경로인 `learn-mcp`, `learn-agent-skills`를 제공합니다. 레슨 본문은 저장소를 클론하지 않아도 이 저장소에서 바로 받아 올 수 있습니다. 다만 저장소의 코드를 그대로 복사해 실행하는 명령과, MCP나 Agent Skills의 실행형 실습에는 로컬 클론이 필요합니다. 진행 상황은 프로젝트 안의 `LEARNING.md`나 `MCP-LEARNING.md`, `AGENT-SKILLS-LEARNING.md`에 저장되므로, 다음 세션에서 그대로 이어서 학습할 수 있습니다.',
        '**Option B — read.** Open any completed lesson on [aiengineeringfromscratch.com](https://aiengineeringfromscratch.com) or expand a phase under [Contents](#contents). No setup, no cloning.':
            '**선택지 B: 읽기.** 완성된 레슨은 [aiengineeringfromscratch.com](https://aiengineeringfromscratch.com)에서 아무거나 열어 보거나, [목차](#contents)에서 원하는 페이즈를 펼쳐 보십시오. 설치할 것도 없고 클론할 것도 없습니다.',
        '**Option C — clone and run.**':
            '**선택지 C: 클론해서 직접 실행하기.**',
        "Cloning also auto-loads the learning skills in Claude Code, and gives every lesson's code to the `learn` tutor for real execution instead of read-along.":
            '저장소를 클론하면 Claude Code에서 학습 스킬이 자동으로 로드되고, `learn` 튜터가 모든 레슨의 코드를 직접 실행할 수 있게 됩니다. 눈으로 따라 읽기만 하는 것과는 다릅니다.',
        'Prepare for Claude certifications':
            'Claude 자격 시험 준비하기',
        'The [Claude Certification Academy](certifications/claude/README.md) is a free, open-source preparation program for all four official Claude certification tracks: Associate Foundations, Developer Foundations, Architect Foundations, and Architect Professional. Each route combines blueprint-mapped lessons, runnable labs, a diagnostic, capstone work, and a full-length original practice exam.':
            '[Claude Certification Academy](certifications/claude/README.md)는 공식 Claude 자격 과정 네 가지를 모두 준비할 수 있는 무료 오픈 소스 프로그램입니다. 네 과정은 Associate Foundations, Developer Foundations, Architect Foundations, Architect Professional입니다. 각 경로는 시험 출제 범위에 대응시킨 레슨, 실행 가능한 실습, 진단 평가, 캡스톤 과제, 그리고 자체 제작한 전체 분량 모의고사로 구성되어 있습니다.',
        'Use the [AI-native GitHub onboarding guide](certifications/claude/GETTING_STARTED.md) with Claude Code, Codex, ChatGPT, Cursor, or another agent. Run `claude-certification` in Codex, `/claude-certification` in Claude Code, or ask another host to use `claude-certification`. It chooses a track, creates a persistent route in `CLAUDE-CERTIFICATION.md`, teaches one step at a time, runs the real labs, and gives artifact-based feedback. The same curriculum remains available on the [certification website](https://aiengineeringfromscratch.com/certifications.html).':
            '[AI 환경에 맞춘 GitHub 온보딩 안내](certifications/claude/GETTING_STARTED.md)를 Claude Code, Codex, ChatGPT, Cursor를 비롯한 에이전트와 함께 사용하십시오. Codex에서는 `claude-certification`을 입력하고, Claude Code에서는 `/claude-certification`을 입력하며, 다른 호스트에서는 `claude-certification`을 사용해 달라고 요청하면 됩니다. 그러면 준비할 과정을 고르고, `CLAUDE-CERTIFICATION.md`에 학습 경로를 저장한 뒤, 한 단계씩 가르치고, 실제 실습을 실행하고, 여러분이 만든 결과물을 근거로 피드백을 줍니다. 같은 커리큘럼을 [자격 과정 웹사이트](https://aiengineeringfromscratch.com/certifications.html)에서도 볼 수 있습니다.',
        'The academy is independent study material based on public exam objectives. It is not affiliated with Anthropic, does not reproduce live exam questions, and cannot guarantee a passing score.':
            '이 학습 과정은 공개된 시험 목표를 근거로 만든 독립적인 학습 자료입니다. Anthropic과는 아무 관계가 없고, 실제 시험 문제를 그대로 싣지 않으며, 합격을 보장하지도 않습니다.',
        'The learning skills':
            '학습 스킬',
        'Read the core curriculum as a book':
            '핵심 커리큘럼을 책으로 읽기',
        'The 20-phase core curriculum under `phases/` compiles into a six-volume book series. EPUB and PDF are built by CI from the same core lesson sources and attached to every [GitHub release](https://github.com/rohitg00/ai-engineering-from-scratch/releases); the links below always resolve to the newest release. Volume numbers index the series, not versions: each copy carries a dated edition stamp, and older editions stay downloadable from their release.':
            '`phases/` 아래에 있는 20개 페이즈 핵심 커리큘럼은 여섯 권짜리 책으로 묶여 나옵니다. EPUB 판본과 PDF 판본은 CI가 같은 레슨 원본에서 만들어 [GitHub 릴리스](https://github.com/rohitg00/ai-engineering-from-scratch/releases)마다 첨부하며, 아래 링크는 언제나 가장 최신 릴리스를 가리킵니다. 권 번호는 판본 번호가 아니라 책의 순서를 가리킵니다. 각 파일에는 발행 날짜가 찍혀 있고, 예전 판본도 해당 릴리스에서 계속 내려받을 수 있습니다.',
        'Certification curricula are intentionally not converted into the books. Their AI tutor state, runnable labs, interactive figures, diagnostics, and timed mocks remain first-class on GitHub and the website.':
            '자격 과정 커리큘럼은 의도적으로 책에 포함하지 않았습니다. 자격 과정은 AI 튜터가 관리하는 학습 상태, 실행 가능한 실습, 상호작용하는 그림, 진단 평가, 시간 제한이 있는 모의고사를 그대로 살려야 하고, 이 요소들은 GitHub과 웹사이트에서만 온전히 동작하기 때문입니다.',
        "The book is the snapshot; this repository is the living edition. Every chapter ends with links back to the lesson's animated figures, quiz, and runnable code. Build locally with `python3 scripts/build_book.py` (pandoc required); pipeline details in [book/README.md](book/README.md).":
            '책은 한 시점을 찍어 둔 판본이고, 이 저장소는 계속 갱신되는 판본입니다. 모든 장의 끝에는 해당 레슨의 애니메이션 그림과 퀴즈, 실행 가능한 코드로 돌아가는 링크가 붙어 있습니다. 직접 만들어 보려면 `python3 scripts/build_book.py`를 실행하십시오. pandoc이 설치되어 있어야 합니다. 빌드 과정의 자세한 내용은 [book/README.md](book/README.md)에 있습니다.',
        'Other curricula end with *"congratulations, you learned X."* Each lesson here ends with a **reusable tool** you can install or paste into your daily workflow.':
            '다른 커리큘럼은 *"축하합니다, X를 배웠습니다"* 라는 말로 끝납니다. 이 커리큘럼의 모든 레슨은 **다시 쓸 수 있는 도구**를 남기며, 그 도구는 여러분의 평소 작업에 바로 설치하거나 붙여넣을 수 있습니다.',
        'Install the lot with `python3 scripts/install_skills.py <target>`. Real tools, not homework. By the end of the curriculum, you have a portfolio of 523 artifacts you actually understand because you built them.':
            '`python3 scripts/install_skills.py <target>` 명령으로 전부 설치할 수 있습니다. 숙제가 아니라 실제로 쓰는 도구입니다. 커리큘럼을 끝낼 즈음이면 결과물 523개가 모인 포트폴리오가 생기고, 전부 직접 만든 것이므로 하나하나 무엇인지 설명할 수 있습니다.',
        'FIG_002 · A worked sample':
            'FIG_002 · 실제 예제 하나',
        'Phase 14, lesson 1: the agent loop. ~120 lines of pure Python, no dependencies.':
            '페이즈 14의 첫 번째 레슨은 에이전트 루프를 다룹니다. 외부 의존성 없이 순수 Python으로 약 120줄입니다.',
        '**`code/agent_loop.py`** &nbsp; <sub><i>build it</i></sub>':
            '**`code/agent_loop.py`** &nbsp; <sub><i>직접 만든다</i></sub>',
        '**`outputs/skill-agent-loop.md`** &nbsp; <sub><i>ship it</i></sub>':
            '**`outputs/skill-agent-loop.md`** &nbsp; <sub><i>결과물로 남긴다</i></sub>',
        '**`outputs/prompt-debug-agent.md`**':
            '**`outputs/prompt-debug-agent.md`**',
        'Twenty phases. Click any phase to expand its lesson list.':
            '20개 페이즈입니다. 페이즈를 누르면 그 안의 레슨 목록이 펼쳐집니다.',
        'Phase 0: Setup & Tooling `12 lessons`':
            '페이즈 0: 환경 구성과 도구 `12개 레슨`',
        'Get your environment ready for everything that follows.':
            '이후에 이어지는 모든 내용을 실행할 수 있도록 개발 환경을 갖춥니다.',
        'Lessons 06-18 and 28-31 form the focused [Model Context Protocol (MCP) path](learning-paths/model-context-protocol.json). Its manifest order is 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 18, 17, 28, 29, 30, 31. Start it with the host-specific `learn-mcp` invocation above. Lesson 23 is its only optional capstone and also requires Lessons 19 and 20.':
            '06번부터 18번까지, 그리고 28번부터 31번까지의 레슨이 집중 경로인 [Model Context Protocol(MCP) 경로](learning-paths/model-context-protocol.json)를 이룹니다. 매니페스트가 정한 순서는 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 18, 17, 28, 29, 30, 31입니다. 위에서 설명한 호스트별 `learn-mcp` 호출 방법으로 시작하십시오. 23번 레슨은 이 경로의 유일한 선택 캡스톤이며, 19번과 20번 레슨도 함께 마쳐야 합니다.',
        'Lessons 22 and 24-27 form the focused [Agent Skills learning path](learning-paths/agent-skills.json), from package contract through real-host release gates. Start it with the host-specific `learn-agent-skills` invocation shown above; do not follow numeric next navigation from 22 to 23.':
            '22번 레슨과 24번부터 27번까지의 레슨이 집중 경로인 [Agent Skills 학습 경로](learning-paths/agent-skills.json)를 이룹니다. 이 경로는 패키지 계약을 정의하는 일에서 출발해 실제 호스트에 배포하기 전의 검증 관문까지 이어집니다. 위에 나온 호스트별 `learn-agent-skills` 호출 방법으로 시작하십시오. 번호 순서를 따라 22번에서 23번으로 넘어가서는 안 됩니다.',
        'Each Phase 14 workbench lesson (31-42) ships a `mission.md` briefing the agent before it opens the full lesson docs.':
            '페이즈 14의 워크벤치 레슨(31번부터 42번까지)에는 각각 `mission.md`가 들어 있습니다. 에이전트는 레슨 문서 전체를 열기 전에 이 파일을 먼저 읽고 과제를 파악합니다.',
        'Lessons 31-46 form the [Agent-Assisted Engineering path](learning-paths/using-coding-agents.json). Its manifest order combines the workbench foundation with task framing, planning, delegation, and durable feedback. Lessons 47-54 form the [Product Judgment and Delivery path](learning-paths/shaping-the-build.json), from outcome framing through evidence, risk, scope, measurement, staged release, and feedback ownership.':
            '31번부터 46번까지의 레슨이 [에이전트 협업 엔지니어링 경로](learning-paths/using-coding-agents.json)를 이룹니다. 이 경로의 매니페스트 순서는 워크벤치 기초 위에 과제 정의, 계획 수립, 작업 위임, 지속적인 피드백을 차례로 얹습니다. 47번부터 54번까지의 레슨은 [제품 판단과 전달 경로](learning-paths/shaping-the-build.json)를 이루며, 성과를 정의하는 일에서 출발해 근거 수집, 위험 파악, 범위 결정, 측정, 단계적 출시, 피드백 책임까지 이어집니다.',
        '**Deep-build tracks** — multi-lesson series that build a complete subsystem from scratch.':
            '**심화 구현 트랙**: 여러 레슨에 걸쳐 하나의 완결된 서브시스템을 맨바닥에서 만들어 내는 연속 과정입니다.',
        'Every lesson produces a reusable artifact. By the end you have:':
            '모든 레슨은 다시 쓸 수 있는 결과물을 하나씩 만들어 냅니다. 커리큘럼을 끝내면 다음을 갖게 됩니다.',
        'Plug them into Claude, Cursor, Codex, OpenClaw, Hermes, or any agent that reads a SKILL.md / AGENTS.md directory. Real tools, not homework.':
            '이 결과물들은 Claude, Cursor, Codex, OpenClaw, Hermes를 비롯해 SKILL.md나 AGENTS.md 디렉터리를 읽는 어떤 에이전트에도 그대로 연결할 수 있습니다. 숙제가 아니라 실제로 쓰는 도구입니다.',
        'Install course skills into your agent':
            '커리큘럼 스킬을 에이전트에 설치하기',
        'Two skill sets, two installers:':
            '스킬 묶음이 두 가지이고, 설치 방법도 두 가지입니다.',
        '**The learning skills** (`start-learning`, `learn`, `course-guide`, `learn-mcp`, `learn-agent-skills`, `claude-certification`, `find-your-level`, and `check-understanding`) live under [`skills/`](skills/) and install into a supported skill-capable host with one command. Installation needs Node.js and `npx`, but not a repository clone or Python:':
            '**학습 스킬**(`start-learning`, `learn`, `course-guide`, `learn-mcp`, `learn-agent-skills`, `claude-certification`, `find-your-level`, `check-understanding`)은 [`skills/`](skills/) 아래에 있으며, 스킬을 지원하는 호스트에 명령 한 줄로 설치됩니다. 설치에는 Node.js와 `npx`가 필요하지만, 저장소 클론이나 Python은 필요하지 않습니다.',
        '`skills` writes to the host and scope selected during installation, such as `.claude/skills/`, `.cursor/skills/`, `.codex/skills/`, or another supported skills folder. Verify that the selected host discovers that exact destination.':
            '`skills` 명령은 설치할 때 선택한 호스트와 범위에 파일을 씁니다. 예를 들면 `.claude/skills/`나 `.cursor/skills/`, `.codex/skills/`, 또는 지원되는 다른 스킬 폴더입니다. 선택한 호스트가 바로 그 경로를 실제로 인식하는지 확인하십시오.',
        '**The lesson artifacts.** The repo ships 396 skills and 99 prompts under `phases/**/outputs/`; install them via `scripts/install_skills.py`. Requires cloning the repo. Supports tag filters, dry-runs, and per-agent layouts:':
            '**레슨 결과물.** 이 저장소는 `phases/**/outputs/` 아래에 스킬 396개와 프롬프트 99개를 담고 있으며, `scripts/install_skills.py`로 설치합니다. 이 방법은 저장소를 클론해야 합니다. 태그로 걸러 내기, 실제로 쓰지 않고 미리 확인하기, 에이전트별 디렉터리 구조로 배치하기를 지원합니다.',
        '`<target>` is the skills directory for your agent (examples: `~/.claude/skills/`, `~/.cursor/skills/`, `~/.config/openclaw/skills/`, `.skills/`, or any path your agent reads).':
            '`<target>`에는 사용하는 에이전트의 스킬 디렉터리를 적습니다. 예를 들면 `~/.claude/skills/`, `~/.cursor/skills/`, `~/.config/openclaw/skills/`, `.skills/` 같은 경로이며, 에이전트가 읽는 경로라면 무엇이든 됩니다.',
        'By default the script refuses to overwrite an existing destination and exits with code 1 after listing every colliding path. Use `--dry-run` to preview collisions or `--force` to overwrite. Every non-dry-run run writes a `manifest.json` in the target with the full inventory grouped by type and phase. Pick the layout your agent reads:':
            '이 스크립트는 기본적으로 이미 있는 대상 파일을 덮어쓰지 않으며, 충돌하는 경로를 모두 출력한 뒤 종료 코드 1로 끝납니다. 충돌을 미리 확인하려면 `--dry-run`을 쓰고, 덮어쓰려면 `--force`를 쓰십시오. 미리 확인하는 경우가 아니라면 실행할 때마다 대상 디렉터리에 `manifest.json`을 남기며, 이 파일에는 설치된 항목 전체가 유형별과 페이즈별로 정리되어 있습니다. 사용하는 에이전트가 읽는 디렉터리 구조를 고르십시오.',
        'Drop the agent workbench into your own repo':
            '에이전트 워크벤치를 여러분의 저장소에 설치하기',
        'The Phase 14 capstone ships a reusable Agent Workbench pack (AGENTS.md, schemas, init / verify / handoff scripts). Scaffold it into any repo with:':
            '페이즈 14의 캡스톤은 다시 쓸 수 있는 Agent Workbench 묶음을 제공합니다. 여기에는 AGENTS.md와 스키마, 그리고 초기화·검증·인계 스크립트가 들어 있습니다. 아래 명령으로 어떤 저장소에든 설치할 수 있습니다.',
        'You get the seven workbench surfaces wired up, a starter `task_board.json`, and a fresh `agent_state.json` at `schema_version: 1`. From there: edit the task, edit `AGENTS.md`, run `scripts/init_agent.py`, hand the contract to your agent. The pack source lives at `phases/14-agent-engineering/42-agent-workbench-capstone/outputs/agent-workbench-pack/`.':
            '설치하면 워크벤치를 이루는 일곱 개 요소가 서로 연결된 상태로 생기고, 시작용 `task_board.json`과 `schema_version: 1`인 새 `agent_state.json`도 함께 만들어집니다. 그다음에는 과제 내용을 수정하고, `AGENTS.md`를 수정하고, `scripts/init_agent.py`를 실행한 뒤, 완성된 계약을 에이전트에 넘기면 됩니다. 이 묶음의 원본은 `phases/14-agent-engineering/42-agent-workbench-capstone/outputs/agent-workbench-pack/`에 있습니다.',
        'Browse the entire course as JSON':
            '커리큘럼 전체를 JSON으로 살펴보기',
        '`scripts/build_catalog.py` walks every phase, every lesson, every artifact on disk and writes `catalog.json` at the repo root. One file, every course truth.':
            '`scripts/build_catalog.py`는 디스크에 있는 모든 페이즈와 모든 레슨, 모든 결과물을 훑어서 저장소 최상위에 `catalog.json`을 씁니다. 커리큘럼의 모든 사실이 이 파일 하나에 담깁니다.',
        'The catalog is filesystem-derived, not README-derived, so counts always match what is actually on disk. Use it for site builds, downstream tooling, or to verify the README counts have not drifted. Schema is documented at the top of the script.':
            '이 카탈로그는 README가 아니라 파일 시스템에서 만들어지므로, 집계된 개수가 언제나 디스크에 실제로 있는 내용과 일치합니다. 사이트를 빌드하거나 후속 도구를 만들 때, 또는 README에 적힌 개수가 어긋나지 않았는지 확인할 때 사용하십시오. 스키마는 스크립트 맨 위에 설명되어 있습니다.',
        'A GitHub Action (`.github/workflows/curriculum.yml`) rebuilds `catalog.json` on every PR and fails the build if the committed file is stale. After editing any lesson, run `python3 scripts/build_catalog.py` and commit the result, or CI will reject the PR. The same workflow runs `audit_lessons.py` in warn-only mode (so existing drift does not block contributors).':
            'GitHub Action(`.github/workflows/curriculum.yml`)이 모든 풀 리퀘스트에서 `catalog.json`을 다시 만들며, 커밋된 파일이 최신 상태가 아니면 빌드를 실패시킵니다. 레슨을 수정했다면 `python3 scripts/build_catalog.py`를 실행하고 그 결과를 커밋하십시오. 그렇게 하지 않으면 CI가 풀 리퀘스트를 거부합니다. 같은 워크플로가 `audit_lessons.py`도 실행하지만, 경고만 출력하는 모드로 실행하므로 이미 존재하는 불일치가 기여자를 막지는 않습니다.',
        "Smoke-check every lesson's Python code":
            '모든 레슨의 Python 코드를 빠르게 점검하기',
        "`scripts/lesson_run.py` byte-compiles every `.py` file under each lesson's `code/` directory. Default mode is syntax-check only — no execution, no API keys, no heavy ML deps required. Catches the regressions contributors introduce most often (bad indentation, broken f-strings, stray edits).":
            '`scripts/lesson_run.py`는 각 레슨의 `code/` 디렉터리에 있는 모든 `.py` 파일을 바이트코드로 컴파일해 봅니다. 기본 모드는 문법 검사만 수행합니다. 코드를 실행하지 않고, API 키도 필요 없으며, 무거운 머신러닝 의존성도 설치할 필요가 없습니다. 기여자가 가장 자주 일으키는 문제인 들여쓰기 오류, 깨진 f-string, 실수로 남은 수정 자국을 잡아냅니다.',
        "`--execute` runs each lesson's `code/main.py` (or the first `.py` file) with a 10-second timeout. Lessons whose entry file starts with a `# requires: pkg1, pkg2` comment listing non-stdlib deps are skipped with reason `needs <deps>`. The script is opt-in and not wired into CI.":
            '`--execute` 옵션을 주면 각 레슨의 `code/main.py`를, 그 파일이 없으면 첫 번째 `.py` 파일을 10초 제한을 두고 실행합니다. 진입 파일 첫 줄에 표준 라이브러리 밖의 의존성을 적은 `# requires: pkg1, pkg2` 주석이 있으면, 그 레슨은 `needs <deps>`라는 이유와 함께 건너뜁니다. 이 옵션은 직접 지정해야 동작하며 CI에는 연결되어 있지 않습니다.',
        'Stdlib only, Python 3.10+. Set `LINK_CHECK_SKIP=domain1,domain2` to override the default skip-list (`twitter.com`, `x.com`, `linkedin.com`, `instagram.com`, `medium.com` — domains that aggressively block automated HEAD/GET).':
            '표준 라이브러리만 사용하며, Python 3.10 이상이 필요합니다. 기본 제외 목록을 바꾸려면 `LINK_CHECK_SKIP=domain1,domain2`를 설정하십시오. 기본값으로 제외되는 도메인은 `twitter.com`, `x.com`, `linkedin.com`, `instagram.com`, `medium.com`이며, 모두 자동화된 HEAD 요청과 GET 요청을 강하게 차단하는 곳입니다.',
        '*"The hottest new programming language is English."*<br/> — **Andrej Karpathy** ([tweet](https://x.com/karpathy/status/1617979122625712128))':
            '*"요즘 가장 뜨거운 새 프로그래밍 언어는 영어다."*<br/> — **Andrej Karpathy** ([트윗](https://x.com/karpathy/status/1617979122625712128))',
        '*"Software engineering is being remade in front of our eyes."*<br/> — **Boris Cherny**, creator of Claude Code':
            '*"소프트웨어 엔지니어링이 우리 눈앞에서 새로 만들어지고 있다."*<br/> — **Boris Cherny**, Claude Code를 만든 사람',
        '*"Models will keep getting better. The skill that compounds is **knowing what to build**."*<br/> — Industry consensus, 2026':
            '*"모델은 앞으로도 계속 좋아진다. 쌓이는 실력은 **무엇을 만들지 아는 능력**이다."*<br/> — 2026년 업계의 공통된 견해',
        'Before submitting a lesson, run the invariant check:':
            '레슨을 제출하기 전에 규칙 검사를 실행하십시오.',
        'Exit code is non-zero when any rule fails. Rules (L001–L010) validate directory shape, `docs/en.md` presence + H1, `code/` non-emptiness, `quiz.json` schema (rejects the legacy `q/choices/answer` keys that caused issue #102), and relative links inside lesson docs.':
            '규칙 가운데 하나라도 실패하면 종료 코드가 0이 아닌 값이 됩니다. L001번부터 L010번까지의 규칙은 디렉터리 구조, `docs/en.md`의 존재 여부와 H1 제목, `code/` 디렉터리가 비어 있지 않은지, `quiz.json`의 스키마, 그리고 레슨 문서 안의 상대 경로 링크를 검사합니다. `quiz.json` 검사는 이슈 #102를 일으켰던 옛 `q`/`choices`/`answer` 키를 거부합니다.',
        'If this manual helped you, star the repo. It keeps the project alive.':
            '이 교재가 도움이 되었다면 저장소에 스타를 눌러 주십시오. 그것이 이 프로젝트를 계속 살아 있게 합니다.',
    },
    "hi": {
        HERO1: "**84% छात्र पहले से ही AI टूल इस्तेमाल करते हैं। पर केवल 18% ही उन्हें पेशेवर रूप से इस्तेमाल करने के लिए तैयार महसूस करते हैं।** यह पाठ्यक्रम इसी खाई को पाटता है।",
        HERO2: "523 पाठ। 20 चरण। ~342 घंटे। Python, TypeScript, Rust, Julia। हर पाठ एक पुन: उपयोग योग्य कलाकृति देता है: एक प्रॉम्प्ट, एक स्किल, एक एजेंट, एक MCP सर्वर। मुफ़्त, ओपन सोर्स, MIT।",
        HERO3: "आप केवल AI सीखते नहीं। आप उसे बनाते हैं। शुरू से अंत तक। अपने हाथों से।",
        WAYS: "शुरू करने के तीन तरीके। कोई एक चुनें।",
        LICENSE_LINE: "MIT। जैसे चाहें इस्तेमाल करें: फ़ोर्क करें, पढ़ाएँ, बेचें, प्रकाशित करें। श्रेय देना अच्छा है, पर ज़रूरी नहीं।",
        MAINTAINED: "[Rohit Ghumare](https://github.com/rohitg00) और समुदाय द्वारा अनुरक्षित।",
        H_HOW: "यह कैसे काम करता है", H_CURR: "पाठ्यक्रम की संरचना", H_LESSON: "एक पाठ की संरचना",
        H_START: "शुरुआत करें", H_PREREQ: "आवश्यक शर्तें", H_BOOK: "इसे किताब की तरह पढ़ें",
        H_SHIPS: "हर पाठ कुछ न कुछ देता है", H_CONTENTS: "विषय-सूची", H_TOOLKIT: "टूलकिट",
        H_WHERE: "कहाँ से शुरू करें", H_WHY: "यह अभी क्यों मायने रखता है", H_CONTRIB: "योगदान करें",
        H_SPONSOR: "काम को प्रायोजित करें", H_STAR: "स्टार इतिहास", H_LICENSE: "लाइसेंस",
    },
    "ar": {
        HERO1: "**\u200f84% من الطلاب يستخدمون أدوات الذكاء الاصطناعي بالفعل، لكن 18% فقط يشعرون بأنهم مستعدون لاستخدامها باحتراف.** هذا المنهج يسدّ هذه الفجوة.",
        HERO2: "\u200f523 دروس. 20 مرحلة. نحو 342 ساعة. Python وTypeScript وRust وJulia. كل درس ينتج مخرجًا قابلًا لإعادة الاستخدام: موجّهًا، أو مهارة، أو وكيلًا، أو خادم MCP. مجاني، مفتوح المصدر، برخصة MIT.",
        HERO3: "أنت لا تتعلّم الذكاء الاصطناعي فحسب، بل تبنيه بنفسك. من البداية إلى النهاية. يدويًا.",
        WAYS: "ثلاث طرق للبدء. اختر واحدة.",
        LICENSE_LINE: "رخصة MIT. استخدمه كما تشاء: انسخه، وعلّمه، وبِعه، وانشره. ذكر المصدر محلّ تقدير، لكنه غير مطلوب.",
        MAINTAINED: "يتولّى صيانته [Rohit Ghumare](https://github.com/rohitg00) والمجتمع.",
        H_HOW: "كيف يعمل هذا", H_CURR: "بنية المنهج", H_LESSON: "بنية الدرس",
        H_START: "البدء", H_PREREQ: "المتطلبات المسبقة", H_BOOK: "اقرأه ككتاب",
        H_SHIPS: "كل درس ينتج شيئًا", H_CONTENTS: "المحتويات", H_TOOLKIT: "مجموعة الأدوات",
        H_WHERE: "من أين تبدأ", H_WHY: "لماذا يهمّ هذا الآن", H_CONTRIB: "المساهمة",
        H_SPONSOR: "ادعم العمل", H_STAR: "سجلّ النجوم", H_LICENSE: "الترخيص",
    },
    "ru": {
        HERO1: "**84% студентов уже используют инструменты ИИ, но лишь 18% чувствуют себя готовыми применять их профессионально.** Этот курс закрывает этот разрыв.",
        HERO2: "523 урока. 20 фаз. ~342 часа. Python, TypeScript, Rust, Julia. Каждый урок оставляет переиспользуемый артефакт: промпт, навык, агент, сервер MCP. Бесплатно, открытый исходный код, лицензия MIT.",
        HERO3: "Вы не просто изучаете ИИ. Вы строите его. От начала до конца. Своими руками.",
        H_START_BUILD: "Начните здесь: выберите, что хотите создать",
        START_BUILD: "Перед началом не нужно просматривать все 523 урока. Выберите одну цель. Каждая ссылка открывает один и тот же курс на GitHub или сайте, и обе версии используют один и тот же код уроков.",
        "| Your goal | Learn on GitHub | Learn on the website |": "| Ваша цель | Учиться на GitHub | Учиться на сайте |",
        "| I am new and want the complete foundation | [Phase 0: Setup and Tooling](phases/00-setup-and-tooling/) | [Dev Environment](https://aiengineeringfromscratch.com/lesson?path=phases/00-setup-and-tooling/01-dev-environment) |": "| Я начинаю и хочу получить полную базу | [Фаза 0: Настройка и инструменты](phases/00-setup-and-tooling/) | [Среда разработки](https://aiengineeringfromscratch.com/lesson?path=phases/00-setup-and-tooling/01-dev-environment) |",
        "| I know Python and want math plus ML foundations | [Phase 1: Math Foundations](phases/01-math-foundations/) | [Linear Algebra Intuition](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/01-linear-algebra-intuition) |": "| Я знаю Python и хочу освоить математику и основы ML | [Фаза 1: Математические основы](phases/01-math-foundations/) | [Интуиция линейной алгебры](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/01-linear-algebra-intuition) |",
        "| I want to build production LLM applications | [Phase 11: LLM Engineering](phases/11-llm-engineering/) | [Prompt Engineering](https://aiengineeringfromscratch.com/lesson?path=phases/11-llm-engineering/01-prompt-engineering) |": "| Я хочу создавать промышленные приложения на LLM | [Фаза 11: Инженерия LLM](phases/11-llm-engineering/) | [Инженерия промптов](https://aiengineeringfromscratch.com/lesson?path=phases/11-llm-engineering/01-prompt-engineering) |",
        "| I want to build agents | [Phase 14: Agent Engineering](phases/14-agent-engineering/) | [The Agent Loop](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/01-the-agent-loop) |": "| Я хочу создавать агентов | [Фаза 14: Инженерия агентов](phases/14-agent-engineering/) | [Цикл агента](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/01-the-agent-loop) |",
        "| I want to use coding agents on real repositories | [Agent-Assisted Engineering path](learning-paths/using-coding-agents.json) | [Agent-Assisted Engineering](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/31-agent-workbench-why-models-fail&learningPath=using-coding-agents) |": "| Я хочу использовать агентов программирования в реальных репозиториях | [Маршрут инженерии с агентами](learning-paths/using-coding-agents.json) | [Инженерия с агентами](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/31-agent-workbench-why-models-fail&learningPath=using-coding-agents) |",
        "| I want to shape the right build before implementation | [Product Judgment and Delivery path](learning-paths/shaping-the-build.json) | [Product Judgment and Delivery](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/47-outcomes-before-output&learningPath=shaping-the-build) |": "| Я хочу определить правильное решение до реализации | [Маршрут продуктовых решений и поставки](learning-paths/shaping-the-build.json) | [Продуктовые решения и поставка](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/47-outcomes-before-output&learningPath=shaping-the-build) |",
        "| I want to build with Model Context Protocol (MCP) | [Model Context Protocol (MCP) route](phases/13-tools-and-protocols/README.md#model-context-protocol-mcp-path) | [Model Context Protocol (MCP) path](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/06-mcp-fundamentals&learningPath=model-context-protocol) |": "| Я хочу разрабатывать с Model Context Protocol (MCP) | [Маршрут Model Context Protocol (MCP)](phases/13-tools-and-protocols/README.md#model-context-protocol-mcp-path) | [Маршрут Model Context Protocol (MCP)](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/06-mcp-fundamentals&learningPath=model-context-protocol) |",
        "| I want to write and ship Agent Skills | [Focused Agent Skills route](phases/13-tools-and-protocols/README.md#agent-skills-fast-path) | [Agent Skills path](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/22-skills-and-agent-sdks&learningPath=agent-skills) |": "| Я хочу писать и выпускать Agent Skills | [Сфокусированный маршрут Agent Skills](phases/13-tools-and-protocols/README.md#agent-skills-fast-path) | [Маршрут Agent Skills](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/22-skills-and-agent-sdks&learningPath=agent-skills) |",
        "| I want to prepare for a Claude certification | [Certification onboarding](certifications/claude/GETTING_STARTED.md) | [Certification Academy](https://aiengineeringfromscratch.com/certifications.html) |": "| Я хочу подготовиться к сертификации Claude | [Начало подготовки](certifications/claude/GETTING_STARTED.md) | [Академия сертификации](https://aiengineeringfromscratch.com/certifications.html) |",
        NOT_SURE: "Не знаете, что выбрать? Используйте [наставника `start-learning` для определения уровня](skills/start-learning/SKILL.md) или [руководство по предварительным требованиям на сайте](https://aiengineeringfromscratch.com/prereqs.html).",
        LEARNING_PATHS: "Сравните четыре основных направления и шесть карьерных маршрутов в [учебных маршрутах по AI Engineering](https://aiengineeringfromscratch.com/learning-paths.html).",
        H_USE_LESSON: "Проходите каждый урок одинаково",
        "1. **Read** `docs/en.md` and explain the core idea in your own words.": "1. **Прочитайте** `docs/en.md` и объясните основную идею своими словами.",
        "2. **Type and build** the important code instead of treating the code block as decoration.": "2. **Наберите и соберите** важный код, а не воспринимайте блок кода как иллюстрацию.",
        "3. **Run** the lesson command from the repository root, the directory containing `README.md` and `phases/`.": "3. **Запустите** команду урока из корня репозитория, где находятся `README.md` и `phases/`.",
        "4. **Keep evidence**: the command, working directory, exit code, meaningful output, and the artifact you changed or produced.": "4. **Сохраните доказательства**: команду, рабочий каталог, код выхода, значимый вывод и изменённый или созданный артефакт.",
        "5. **Continue** only when you can explain the output and make one small change without guessing.": "5. **Продолжайте** только тогда, когда можете объяснить вывод и внести небольшое изменение без догадок.",
        LESSON_COMMANDS: "Команды на страницах уроков используют пути от корня репозитория, если урок явно не требует перейти в другой каталог. Если доступно несколько языков, запускайте реализацию для языка, который изучаете.",
        H_CLONE_EVIDENCE: "Клонируйте репозиторий и получите первое доказательство",
        PREFLIGHT: "Предварительная проверка отделяет требования, нужные сейчас, от инструментов, которые понадобятся позже. Для каждой обязательной ошибки показаны обнаруженная причина и команда исправления. Вторая команда запускает урок без зависимостей и показывает, что умножение матрицы на вектор является операцией внутри слоя нейронной сети. Сохраните этот вывод терминала как первое доказательство.",
        WAYS: "Три способа начать. Выберите один.",
        LICENSE_LINE: "MIT. Используйте как угодно: форкайте, преподавайте, продавайте, публикуйте. Указание авторства приветствуется, но не обязательно.",
        MAINTAINED: "Поддерживается [Rohit Ghumare](https://github.com/rohitg00) и сообществом.",
        H_HOW: "Как это устроено", H_CURR: "Структура курса", H_LESSON: "Структура урока",
        H_START: "Начало работы", H_PREREQ: "Предварительные требования", H_BOOK: "Читать как книгу",
        H_SHIPS: "Каждый урок что-то даёт", H_CONTENTS: "Содержание", H_TOOLKIT: "Набор инструментов",
        H_WHERE: "С чего начать", H_WHY: "Почему это важно сейчас", H_CONTRIB: "Как внести вклад",
        H_SPONSOR: "Поддержать проект", H_STAR: "История звёзд", H_LICENSE: "Лицензия",
    },
    "tr": {
        HERO1: "**Öğrencilerin %84'ü zaten yapay zeka araçlarını kullanıyor, ama yalnızca %18'i bunları profesyonelce kullanmaya hazır hissediyor.** Bu müfredat bu boşluğu kapatır.",
        HERO2: "523 ders. 20 aşama. ~342 saat. Python, TypeScript, Rust, Julia. Her ders yeniden kullanılabilir bir çıktı verir: bir istem, bir beceri, bir ajan, bir MCP sunucusu. Ücretsiz, açık kaynak, MIT.",
        HERO3: "Yapay zekayı yalnızca öğrenmezsiniz. Onu kendiniz kurarsınız. Baştan sona. Elle.",
        WAYS: "Başlamanın üç yolu. Birini seçin.",
        LICENSE_LINE: "MIT. İstediğiniz gibi kullanın: çatallayın, öğretin, satın, yayımlayın. Atıf makbule geçer ama zorunlu değildir.",
        MAINTAINED: "[Rohit Ghumare](https://github.com/rohitg00) ve topluluk tarafından sürdürülmektedir.",
        H_HOW: "Nasıl çalışır", H_CURR: "Müfredatın yapısı", H_LESSON: "Bir dersin yapısı",
        H_START: "Başlarken", H_PREREQ: "Ön koşullar", H_BOOK: "Kitap olarak okuyun",
        H_SHIPS: "Her ders bir şey üretir", H_CONTENTS: "İçindekiler", H_TOOLKIT: "Araç seti",
        H_WHERE: "Nereden başlamalı", H_WHY: "Bu neden şimdi önemli", H_CONTRIB: "Katkıda bulunma",
        H_SPONSOR: "Projeye sponsor olun", H_STAR: "Yıldız geçmişi", H_LICENSE: "Lisans",
    },
}

SPONSOR_TRANSLATIONS = {
    "es": {
        H_SPONSORS: "Patrocinadores",
        SPONSOR_ALT: "SerpApi. API de búsqueda web para tus aplicaciones de IA. Disponible en Markdown y JSON para cualquier integración.",
        SPONSOR_THANKS: "Gracias a nuestros patrocinadores.",
        SPONSOR_SUPPORT: "Tu apoyo mantiene cada lección gratuita y de código abierto.",
        SEE_SUPPORTERS: "Ver todos los colaboradores",
        SPONSOR_CLOSING: "Gratis, con licencia MIT, 523 lecciones. Gracias a los patrocinadores y colaboradores que hacen posible este trabajo. [Ver todos los patrocinadores y colaboradores](BACKERS.md).",
        SPONSOR_INVITE: "¿Quieres apoyar el proyecto? Consulta las [opciones de patrocinio](SPONSORS.md), incluidos los [patrocinios de hardware](SPONSORS.md#hardware-lab-partner), o [patrocina en GitHub](https://github.com/sponsors/rohitg00).",
    },
    "fr": {
        H_SPONSORS: "Partenaires",
        SPONSOR_ALT: "SerpApi. API de recherche Web pour vos applications d’IA. Disponible en Markdown et JSON pour toute intégration.",
        SPONSOR_THANKS: "Merci à nos sponsors.",
        SPONSOR_SUPPORT: "Votre soutien permet à chaque leçon de rester gratuite et open source.",
        SEE_SUPPORTERS: "Voir tous les soutiens",
        SPONSOR_CLOSING: "Gratuit, sous licence MIT, 523 leçons. Merci aux sponsors et aux soutiens qui rendent ce travail possible. [Voir tous les sponsors et soutiens](BACKERS.md).",
        SPONSOR_INVITE: "Vous souhaitez soutenir le projet ? Consultez les [options de sponsoring](SPONSORS.md), notamment le [sponsoring matériel](SPONSORS.md#hardware-lab-partner), ou [soutenez le projet sur GitHub](https://github.com/sponsors/rohitg00).",
    },
    "pt": {
        H_SPONSORS: "Patrocinadores",
        SPONSOR_ALT: "SerpApi. API de busca na Web para seus aplicativos de IA. Disponível em Markdown e JSON para qualquer integração.",
        SPONSOR_THANKS: "Agradecemos aos nossos patrocinadores.",
        SPONSOR_SUPPORT: "Seu apoio mantém todas as lições gratuitas e de código aberto.",
        SEE_SUPPORTERS: "Ver todos os apoiadores",
        SPONSOR_CLOSING: "Grátis, com licença MIT, 523 lições. Agradecemos aos patrocinadores e apoiadores que tornam este trabalho possível. [Ver todos os patrocinadores e apoiadores](BACKERS.md).",
        SPONSOR_INVITE: "Quer apoiar o projeto? Veja as [opções de patrocínio](SPONSORS.md), incluindo [patrocínios de hardware](SPONSORS.md#hardware-lab-partner), ou [patrocine pelo GitHub](https://github.com/sponsors/rohitg00).",
    },
    "de": {
        H_SPONSORS: "Sponsoren",
        SPONSOR_ALT: "SerpApi. Websuch-API für deine KI-Anwendungen. Für jede Integration in Markdown und JSON verfügbar.",
        SPONSOR_THANKS: "Vielen Dank an unsere Sponsoren.",
        SPONSOR_SUPPORT: "Deine Unterstützung hält jede Lektion kostenlos und quelloffen.",
        SEE_SUPPORTERS: "Alle Unterstützer ansehen",
        SPONSOR_CLOSING: "Kostenlos, MIT-lizenziert, 523 Lektionen. Vielen Dank an die Sponsoren und Unterstützer, die diese Arbeit ermöglichen. [Alle Sponsoren und Unterstützer ansehen](BACKERS.md).",
        SPONSOR_INVITE: "Möchtest du die Arbeit unterstützen? Sieh dir die [Sponsoring-Optionen](SPONSORS.md) einschließlich [Hardware-Sponsoring](SPONSORS.md#hardware-lab-partner) an oder [unterstütze das Projekt auf GitHub](https://github.com/sponsors/rohitg00).",
    },
    "it": {
        H_SPONSORS: "Sponsor",
        SPONSOR_ALT: "SerpApi. API di ricerca Web per le tue applicazioni di IA. Disponibile in Markdown e JSON per qualsiasi integrazione.",
        SPONSOR_THANKS: "Grazie ai nostri sponsor.",
        SPONSOR_SUPPORT: "Il tuo sostegno mantiene ogni lezione gratuita e open source.",
        SEE_SUPPORTERS: "Vedi tutti i sostenitori",
        SPONSOR_CLOSING: "Gratuito, con licenza MIT, 523 lezioni. Grazie agli sponsor e ai sostenitori che rendono possibile questo lavoro. [Vedi tutti gli sponsor e i sostenitori](BACKERS.md).",
        SPONSOR_INVITE: "Vuoi sostenere il progetto? Consulta le [opzioni di sponsorizzazione](SPONSORS.md), incluse le [sponsorizzazioni hardware](SPONSORS.md#hardware-lab-partner), oppure [sostienilo su GitHub](https://github.com/sponsors/rohitg00).",
    },
    "zh": {
        H_SPONSORS: "赞助方",
        SPONSOR_ALT: "SerpApi。面向 AI 应用的网页搜索 API，可为任何集成提供 Markdown 和 JSON 格式。",
        SPONSOR_THANKS: "感谢我们的赞助方。",
        SPONSOR_SUPPORT: "你的支持让每节课都能保持免费和开源。",
        SEE_SUPPORTERS: "查看所有支持者",
        SPONSOR_CLOSING: "免费、采用 MIT 许可证，共 523 节课。感谢所有让这项工作成为可能的赞助方和支持者。[查看所有赞助方和支持者](BACKERS.md)。",
        SPONSOR_INVITE: "想支持这项工作？请查看[赞助方案](SPONSORS.md)，包括[硬件赞助](SPONSORS.md#hardware-lab-partner)，或[通过 GitHub 赞助](https://github.com/sponsors/rohitg00)。",
    },
    "ja": {
        H_SPONSORS: "スポンサー",
        SPONSOR_ALT: "SerpApi。AIアプリ向けのWeb検索API。あらゆる連携に使えるMarkdown形式とJSON形式に対応しています。",
        SPONSOR_THANKS: "スポンサーの皆さまに感謝します。",
        SPONSOR_SUPPORT: "皆さまの支援により、すべてのレッスンを無料かつオープンソースで提供できます。",
        SEE_SUPPORTERS: "すべての支援者を見る",
        SPONSOR_CLOSING: "無料、MITライセンス、523レッスン。この取り組みを支えるスポンサーと支援者の皆さまに感謝します。[すべてのスポンサーと支援者を見る](BACKERS.md)。",
        SPONSOR_INVITE: "この取り組みを支援するには、[スポンサーシップの選択肢](SPONSORS.md)と[ハードウェアスポンサーシップ](SPONSORS.md#hardware-lab-partner)をご覧になるか、[GitHubでスポンサーになる](https://github.com/sponsors/rohitg00)ことができます。",
    },
    "ko": {
        H_SPONSORS: "후원사",
        SPONSOR_ALT: "SerpApi. AI 앱을 위한 웹 검색 API. 어떤 통합에도 사용할 수 있도록 Markdown과 JSON으로 제공합니다.",
        SPONSOR_THANKS: "후원사 여러분께 감사드립니다.",
        SPONSOR_SUPPORT: "여러분의 후원으로 모든 레슨을 무료 오픈소스로 유지할 수 있습니다.",
        SEE_SUPPORTERS: "모든 후원자 보기",
        SPONSOR_CLOSING: "무료, MIT 라이선스, 523개 레슨. 이 작업을 가능하게 해 주는 후원사와 후원자 여러분께 감사드립니다. [모든 후원사와 후원자 보기](BACKERS.md).",
        SPONSOR_INVITE: "이 작업을 지원하려면 [후원 옵션](SPONSORS.md)과 [하드웨어 후원](SPONSORS.md#hardware-lab-partner)을 확인하거나 [GitHub에서 후원](https://github.com/sponsors/rohitg00)하세요.",
    },
    "hi": {
        H_SPONSORS: "प्रायोजक",
        SPONSOR_ALT: "SerpApi। आपके AI ऐप्स के लिए वेब खोज API। किसी भी एकीकरण के लिए Markdown और JSON में उपलब्ध।",
        SPONSOR_THANKS: "हमारे प्रायोजकों का धन्यवाद।",
        SPONSOR_SUPPORT: "आपका सहयोग हर पाठ को मुफ़्त और ओपन सोर्स बनाए रखता है।",
        SEE_SUPPORTERS: "सभी समर्थक देखें",
        SPONSOR_CLOSING: "मुफ़्त, MIT लाइसेंस के अंतर्गत, 523 पाठ। इस काम को संभव बनाने वाले प्रायोजकों और समर्थकों का धन्यवाद। [सभी प्रायोजक और समर्थक देखें](BACKERS.md)।",
        SPONSOR_INVITE: "इस काम में सहयोग करना चाहते हैं? [प्रायोजन विकल्प](SPONSORS.md), जिनमें [हार्डवेयर प्रायोजन](SPONSORS.md#hardware-lab-partner) शामिल है, देखें या [GitHub पर प्रायोजित करें](https://github.com/sponsors/rohitg00)।",
    },
    "ar": {
        H_SPONSORS: "الرعاة",
        SPONSOR_ALT: "SerpApi. واجهة API للبحث على الويب لتطبيقات الذكاء الاصطناعي، متاحة بصيغتي Markdown وJSON لأي تكامل.",
        SPONSOR_THANKS: "شكرًا لرعاتنا.",
        SPONSOR_SUPPORT: "دعمكم يُبقي كل درس مجانيًا ومفتوح المصدر.",
        SEE_SUPPORTERS: "عرض جميع الداعمين",
        SPONSOR_CLOSING: "مجاني، بترخيص MIT، ويضم 523 درسًا. شكرًا للرعاة والداعمين الذين يجعلون هذا العمل ممكنًا. [عرض جميع الرعاة والداعمين](BACKERS.md).",
        SPONSOR_INVITE: "هل ترغب في دعم العمل؟ اطّلع على [خيارات الرعاية](SPONSORS.md)، بما فيها [رعاية الأجهزة](SPONSORS.md#hardware-lab-partner)، أو [قدّم رعايتك عبر GitHub](https://github.com/sponsors/rohitg00).",
    },
    "ru": {
        H_SPONSORS: "Спонсоры",
        SPONSOR_ALT: "SerpApi. API веб-поиска для ваших приложений с ИИ. Доступен в форматах Markdown и JSON для любой интеграции.",
        SPONSOR_THANKS: "Спасибо нашим спонсорам.",
        SPONSOR_SUPPORT: "Ваша поддержка помогает сохранять все уроки бесплатными и открытыми.",
        SEE_SUPPORTERS: "Посмотреть всех сторонников",
        SPONSOR_CLOSING: "Бесплатно, по лицензии MIT, 523 урока. Спасибо спонсорам и сторонникам, благодаря которым эта работа возможна. [Посмотреть всех спонсоров и сторонников](BACKERS.md).",
        SPONSOR_INVITE: "Хотите поддержать проект? Посмотрите [варианты спонсорства](SPONSORS.md), включая [спонсорство оборудования](SPONSORS.md#hardware-lab-partner), или [станьте спонсором на GitHub](https://github.com/sponsors/rohitg00).",
    },
    "tr": {
        H_SPONSORS: "Sponsorlar",
        SPONSOR_ALT: "SerpApi. Yapay zeka uygulamalarınız için Web Arama API'si. Her türlü entegrasyon için Markdown ve JSON biçimlerinde sunulur.",
        SPONSOR_THANKS: "Sponsorlarımıza teşekkür ederiz.",
        SPONSOR_SUPPORT: "Desteğiniz her dersin ücretsiz ve açık kaynak kalmasını sağlar.",
        SEE_SUPPORTERS: "Tüm destekçileri görüntüle",
        SPONSOR_CLOSING: "Ücretsiz, MIT lisanslı, 523 ders. Bu çalışmayı mümkün kılan sponsorlara ve destekçilere teşekkür ederiz. [Tüm sponsorları ve destekçileri görüntüle](BACKERS.md).",
        SPONSOR_INVITE: "Çalışmayı desteklemek ister misiniz? [Sponsorluk seçeneklerini](SPONSORS.md), [donanım sponsorluğunu](SPONSORS.md#hardware-lab-partner) inceleyin veya [GitHub üzerinden sponsor olun](https://github.com/sponsors/rohitg00).",
    },
}

for language, sponsor_translations in SPONSOR_TRANSLATIONS.items():
    TRANSLATIONS[language].update(sponsor_translations)
