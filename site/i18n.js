/**
 * AIFS translated UI layer.
 *
 * The site-wide language control is site/lang-picker.js (40 languages, backed
 * by languages.json). This module does not own the language state: it reads
 * whatever the picker selected and supplies hand-written UI labels when a table
 * exists for that language. Every other language falls back to English labels.
 *
 * Adding a language here is one entry in UI keyed by its languages.json `code`.
 * Korean is the first one; nothing below this comment names it.
 *
 * Data-driven content uses pick() against the per-language field suffix baked
 * into data.js by build.js (name → nameKo for 'ko'). A missing translated field
 * falls back to English, so partial coverage renders correctly.
 */
(function () {
  var STORAGE_KEY = 'lang';          // owned by lang-picker.js
  var LEGACY_KEYS = ['siteLang', 'lessonLang'];

  // One-time migration: honor a language chosen by an older per-page toggle.
  try {
    if (!localStorage.getItem(STORAGE_KEY)) {
      for (var m = 0; m < LEGACY_KEYS.length; m++) {
        var legacy = localStorage.getItem(LEGACY_KEYS[m]);
        if (legacy && legacy !== 'en') {
          localStorage.setItem(STORAGE_KEY, legacy);
          break;
        }
      }
    }
  } catch (e) { /* localStorage disabled */ }

  var UI = {
    en: {
      'loading': 'Loading lesson…',
      'learningObjectives': 'Learning Objectives',
      'quiz.pre': 'Pre-Lesson Check', 'quiz.mid': 'Mid-Lesson Check',
      'quiz.post': 'Post-Lesson Quiz', 'quiz.all': 'Quiz',
      'phase': 'Phase',
      'search.placeholder': 'Search…',
      'count.lessons': 'lessons', 'count.phases': 'phases',
      // header nav (shared across pages)
      'nav.contents': 'Contents', 'nav.catalog': 'Catalog',
      'nav.roadmap': 'Roadmap', 'nav.glossary': 'Glossary',
      'nav.books': 'Books', 'nav.about': 'About',
      'nav.certifications': 'Certifications', 'nav.learningPaths': 'Learning Paths',
      // home / index landing copy
      'home.tagline.rest': 'Every algorithm built from raw math before a single framework gets imported.',
      'home.attribution': 'Maintained by Rohit Ghumare and contributors. Run on your own machine.',
      'home.btn.start': 'Start the Course', 'home.btn.paths': 'Explore Learning Paths',
      'home.btn.star': 'Star on GitHub', 'home.btn.follow': 'Follow @rohitg00',
      'home.preface.eyebrow': 'How this works',
      'home.intro1': "Most AI material teaches in scattered pieces. A paper here, a fine-tuning post there, a flashy agent demo somewhere else. The pieces rarely line up. You ship a chatbot but can't explain its loss curve. You hook a function to an agent but can't say what attention does inside the model that's calling it.",
      'home.intro2.head': 'This curriculum is the spine.',
      'home.intro2.rest': "four languages: Python, TypeScript, Rust, Julia. Linear algebra at one end, autonomous swarms at the other. Every algorithm gets built from raw math first. Backprop. Tokenizer. Attention. Agent loop. By the time PyTorch shows up, you already know what it's doing under the hood.",
      'home.intro3': 'Each lesson runs the same loop: read the problem, derive the math, write the code, run the test, keep the artifact. No five-minute videos, no copy-paste deploys, no hand-holding. Free, open source, and built to run on your own laptop.',
      'home.stat.title': 'Current Progress',
      'home.stat.finished': 'Finished Lessons', 'home.stat.phases': 'Phases',
      'home.stat.languages': 'Languages', 'home.stat.glossary': 'Glossary Terms',
      'home.toc.title': 'Curriculum · 20 phases · 523 lessons',
      'home.toc.subtitle': 'Tap a phase to expand its lessons. Each one ships when its math, code, and test are all written.',
      'home.legend.complete': 'Complete', 'home.legend.inprogress': 'In progress', 'home.legend.planned': 'Planned',
      'home.modal.footnote': 'Progress saved in browser only', 'home.modal.reset': 'Reset progress',
      'home.colophon.eyebrow': 'Colophon',
      'home.colophon.body': 'The entire curriculum is on GitHub. Clone it, fork it, learn at your own pace. No paywall, no signup. Every lesson has runnable code in Python, TypeScript, Rust, or Julia, depending on what fits the concept best.',
      'home.footer': '© 2026 · open source · free forever', 'home.footer.report': 'Report',
      // phase modal (home page)
      'modal.open': 'Open lesson', 'modal.review': 'Review',
      'modal.openAria': 'Open lesson', 'modal.comingSoon': 'Coming soon',
      'modal.markDone': 'Mark complete', 'modal.markUndone': 'Mark as not done', 'modal.done': 'Done',
      // shared footer
      'footer.home': 'Home', 'footer.report': 'Report / Suggest',
      'footer.tagline': 'AI Engineering from Scratch · open source · free forever.',
      // catalog
      'catalog.title': 'Lesson Catalog',
      'catalog.subtitle': 'Every lesson across all 20 phases. Search, filter, sort.',
      'catalog.search': 'Search lessons...',
      'catalog.allPhases': 'All Phases', 'catalog.allStatus': 'All Status',
      'catalog.complete': 'Complete', 'catalog.planned': 'Planned', 'catalog.inprogress': 'In progress',
      'catalog.th.phase': 'Phase', 'catalog.th.lesson': 'Lesson', 'catalog.th.type': 'Type',
      'catalog.th.language': 'Language', 'catalog.th.status': 'Status',
      'catalog.empty': 'No lessons match your filters.',
      'catalog.lessons': 'lessons', 'catalog.of': 'of',
      // prereqs / roadmap
      'prereqs.title': 'Roadmap',
      'prereqs.eyebrow': 'Curriculum navigation',
      'prereqs.heading': 'From first principles to production AI.',
      'prereqs.lede': 'Twenty connected phases form one top-to-bottom dependency graph. Follow the route downward, illuminate the prerequisites behind any phase, and continue exactly where your local progress left off.',
      'prereqs.stat.lessons': 'Lessons', 'prereqs.stat.progress': 'Your progress',
      'prereqs.stat.next': 'Recommended next',
      'prereqs.stages.title': 'Four graph zones', 'prereqs.stages.hint': 'Jump down the map',
      'prereqs.skip': 'Skip to content',
      'prereqs.none': 'None. This is a starting point.',
      'prereqs.final': 'Final destination. End of the curriculum.',
      'prereqs.prerequisites': 'Prerequisites', 'prereqs.unlocks': 'Unlocks',
      'prereqs.read': 'Read', 'prereqs.github': 'View on GitHub',
      'prereqs.lessonsComplete': 'lessons complete',
      'prereqs.prereqPhases': 'prerequisite phases', 'prereqs.phasesUnlocked': 'phases unlocked',
      // glossary
      'glossary.title': 'AI Engineering Glossary',
      'glossary.kicker': 'Reference ledger · curriculum v1.0',
      'glossary.deck1': 'Precise working definitions for the systems you build. Start with the meaning, then use the examples, distinctions, and lesson links to turn vocabulary into judgment.',
      'glossary.stat.terms': 'Terms indexed', 'glossary.stat.categories': 'Categories',
      'glossary.search': 'Search terms...',
      'glossary.says': 'What people say', 'glossary.means': 'What it actually means',
      'glossary.terms': 'terms', 'glossary.empty': 'No terms match your search.',
      'glossary.label.definition': 'Working definition',
      'glossary.label.pending': 'Definition in progress.',
      'glossary.label.whyItMatters': 'Why it matters',
      'glossary.label.example': 'In practice',
      'glossary.label.says': 'Common shortcut',
      'glossary.label.confusion': 'Do not confuse it with',
      'glossary.label.whyCalled': 'Why it is called this',
      'glossary.label.lessons': 'Learn it in the course',
      'glossary.label.sources': 'Primary sources',
      'glossary.label.aliases': 'Also called'
    },
    ko: {
      'loading': '레슨을 불러오는 중입니다…',
      'learningObjectives': '학습 목표',
      'quiz.pre': '사전 점검', 'quiz.mid': '중간 점검',
      'quiz.post': '사후 퀴즈', 'quiz.all': '퀴즈',
      'phase': '페이즈',
      'search.placeholder': '검색…',
      'count.lessons': '개 레슨', 'count.phases': '개 페이즈',
      // header nav (shared across pages)
      'nav.contents': '목차', 'nav.catalog': '카탈로그',
      'nav.roadmap': '로드맵', 'nav.glossary': '용어집',
      'nav.books': '도서', 'nav.about': '소개',
      'nav.certifications': '자격 과정', 'nav.learningPaths': '학습 경로',
      // home / index landing copy
      'home.tagline.rest': '단 하나의 프레임워크도 들여오기 전에, 모든 알고리즘을 날것의 수학에서부터 직접 만듭니다.',
      'home.attribution': 'Rohit Ghumare와 기여자들이 관리합니다. 여러분의 컴퓨터에서 직접 실행하십시오.',
      'home.btn.start': '커리큘럼 시작하기', 'home.btn.paths': '학습 경로 둘러보기',
      'home.btn.star': 'GitHub에서 스타 누르기', 'home.btn.follow': '@rohitg00 팔로우하기',
      'home.preface.eyebrow': '이 커리큘럼이 진행되는 방식',
      'home.intro1': 'AI를 다루는 자료는 대부분 흩어진 조각으로 가르칩니다. 여기에는 논문이 하나 있고, 저기에는 파인튜닝(fine-tuning)을 설명하는 글이 하나 있으며, 또 다른 곳에는 화려한 에이전트 데모가 하나 있습니다. 그 조각들은 서로 좀처럼 맞물리지 않습니다. 그래서 챗봇을 출시하고도 그 손실(loss) 곡선이 왜 그렇게 움직이는지 설명하지 못합니다. 에이전트에 함수를 연결해 놓고도, 그 함수를 호출하는 모델 내부에서 어텐션(attention)이 무슨 일을 하는지 말하지 못합니다.',
      'home.intro2.head': '이 커리큘럼은 그 조각들을 하나로 잇는 중심 구조입니다.',
      'home.intro2.rest': '그리고 Python, TypeScript, Rust, Julia까지 네 가지 언어를 사용합니다. 한쪽 끝에는 선형대수(linear algebra)가 있고, 반대쪽 끝에는 자율 군집(autonomous swarm)이 있습니다. 모든 알고리즘은 날것의 수학에서부터 먼저 구현합니다. 역전파(backpropagation)를 구현하고, 토크나이저(tokenizer)를 구현하고, 어텐션을 구현하고, 에이전트 루프(agent loop)를 구현합니다. PyTorch가 등장할 무렵이면, 여러분은 이미 PyTorch가 내부에서 무엇을 수행하는지 알고 있게 됩니다.',
      'home.intro3': '모든 레슨은 동일한 절차를 따릅니다. 문제를 읽고, 수학을 유도하고, 코드를 작성하고, 테스트를 실행하고, 그 결과물을 보관합니다. 5분짜리 영상도 없고, 복사해서 붙여넣는 배포도 없으며, 하나하나 떠먹여 주는 안내도 없습니다. 이 커리큘럼은 무료이고 오픈 소스이며, 여러분의 노트북에서 그대로 실행되도록 만들었습니다.',
      'home.stat.title': '현재 진행 상황',
      'home.stat.finished': '완료한 레슨', 'home.stat.phases': '페이즈',
      'home.stat.languages': '언어', 'home.stat.glossary': '용어집 항목',
      'home.toc.title': '커리큘럼 · 20개 페이즈 · 523개 레슨',
      'home.toc.subtitle': '페이즈를 누르면 그 안의 레슨이 펼쳐집니다. 각 레슨은 수학과 코드와 테스트가 모두 작성되었을 때 공개됩니다.',
      'home.legend.complete': '완료', 'home.legend.inprogress': '진행 중', 'home.legend.planned': '예정',
      'home.modal.footnote': '진행 상황은 브라우저에만 저장됩니다.', 'home.modal.reset': '진행 상황 초기화',
      'home.colophon.eyebrow': '판권',
      'home.colophon.body': '커리큘럼 전체가 GitHub에 공개되어 있습니다. 저장소를 클론하거나 포크해서 각자의 속도로 학습하십시오. 유료 장벽도 없고 가입 절차도 없습니다. 모든 레슨에는 실행할 수 있는 코드가 들어 있으며, 그 코드는 개념을 설명하기에 가장 적합한 언어인 Python, TypeScript, Rust, Julia 가운데 하나로 작성되어 있습니다.',
      'home.footer': '© 2026 · 오픈 소스 · 영원히 무료', 'home.footer.report': '제보하기',
      // phase modal (home page)
      'modal.open': '레슨 열기', 'modal.review': '다시 보기',
      'modal.openAria': '레슨 열기', 'modal.comingSoon': '준비 중입니다',
      'modal.markDone': '완료로 표시', 'modal.markUndone': '완료 표시 해제', 'modal.done': '완료함',
      // shared footer
      'footer.home': '홈', 'footer.report': '제보 / 제안',
      'footer.tagline': 'AI Engineering from Scratch · 오픈 소스 · 영원히 무료입니다.',
      // catalog
      'catalog.title': '레슨 카탈로그',
      'catalog.subtitle': '20개 페이즈에 속한 모든 레슨을 검색하고, 조건으로 거르고, 원하는 기준으로 정렬할 수 있습니다.',
      'catalog.search': '레슨 검색...',
      'catalog.allPhases': '모든 페이즈', 'catalog.allStatus': '모든 상태',
      'catalog.complete': '완료', 'catalog.planned': '예정', 'catalog.inprogress': '진행 중',
      'catalog.th.phase': '페이즈', 'catalog.th.lesson': '레슨', 'catalog.th.type': '유형',
      'catalog.th.language': '언어', 'catalog.th.status': '상태',
      'catalog.empty': '지정한 조건에 해당하는 레슨이 없습니다.',
      'catalog.lessons': '개 레슨', 'catalog.of': '/',
      // prereqs / roadmap
      'prereqs.title': '로드맵',
      'prereqs.eyebrow': '커리큘럼 길잡이',
      'prereqs.heading': '제1원리에서 출발해 실제로 운영되는 AI까지 나아갑니다.',
      'prereqs.lede': '서로 연결된 20개 페이즈가 위에서 아래로 이어지는 하나의 의존 관계 그래프를 이룹니다. 그 경로를 따라 아래로 내려가면서, 각 페이즈의 뒤에 놓인 선행 조건을 확인하고, 이전에 학습을 멈춘 지점에서 그대로 이어서 진행하십시오.',
      'prereqs.stat.lessons': '레슨', 'prereqs.stat.progress': '나의 진행 상황',
      'prereqs.stat.next': '다음 추천 페이즈',
      'prereqs.stages.title': '네 개의 구역', 'prereqs.stages.hint': '지도의 원하는 위치로 이동하십시오',
      'prereqs.skip': '본문으로 건너뛰기',
      'prereqs.none': '선행 조건이 없습니다. 여기가 출발점입니다.',
      'prereqs.final': '최종 목적지이며, 커리큘럼이 여기에서 끝납니다.',
      'prereqs.prerequisites': '선행 조건', 'prereqs.unlocks': '이후에 열리는 페이즈',
      'prereqs.read': '읽기', 'prereqs.github': 'GitHub에서 보기',
      'prereqs.lessonsComplete': '개 레슨 완료',
      'prereqs.prereqPhases': '개 선행 페이즈', 'prereqs.phasesUnlocked': '개 페이즈가 열림',
      // glossary
      'glossary.title': 'AI 엔지니어링 용어집',
      'glossary.kicker': '참조 대장 · 커리큘럼 v1.0',
      'glossary.deck1': '여러분이 직접 만드는 시스템에 그대로 적용할 수 있도록, 각 용어를 정확하게 정의해 두었습니다. 먼저 의미를 읽고, 이어지는 예시와 혼동하기 쉬운 개념의 구분, 그리고 연결된 레슨을 따라가면서 용어를 실제 판단에 쓸 수 있는 지식으로 바꾸십시오.',
      'glossary.stat.terms': '수록된 용어', 'glossary.stat.categories': '분류',
      'glossary.search': '용어 검색...',
      'glossary.says': '사람들이 말하는 것', 'glossary.means': '실제로 의미하는 것',
      'glossary.terms': '개 용어', 'glossary.empty': '검색어에 해당하는 용어가 없습니다.',
      'glossary.label.definition': '실무에서 쓰는 정의',
      'glossary.label.pending': '아직 정의를 작성하는 중입니다.',
      'glossary.label.whyItMatters': '이 개념이 중요한 이유',
      'glossary.label.example': '실제 사용 방식',
      'glossary.label.says': '흔히 줄여서 하는 말',
      'glossary.label.confusion': '혼동하기 쉬운 개념',
      'glossary.label.whyCalled': '이런 이름이 붙은 이유',
      'glossary.label.lessons': '이 커리큘럼에서 배우는 곳',
      'glossary.label.sources': '1차 출처',
      'glossary.label.aliases': '다르게 부르는 이름'
    }
  };

  // A language is usable here only when UI carries a table for it; anything
  // else renders English chrome even though the lesson body may be translated.
  function supported(lang) {
    return !!(lang && Object.prototype.hasOwnProperty.call(UI, lang));
  }

  // The picker owns the language. We only resolve it to a table we can render.
  function getLang() {
    try {
      if (typeof window.AIFS_currentLang === 'function') {
        var picked = window.AIFS_currentLang();
        return supported(picked) ? picked : 'en';
      }
      var params = new URLSearchParams(window.location.search);
      var fromQuery = params.get('lang');
      if (supported(fromQuery)) return fromQuery;
      var stored = localStorage.getItem(STORAGE_KEY);
      if (supported(stored)) return stored;
    } catch (e) { /* ignore */ }
    return 'en';
  }

  // 'ko' → 'Ko', 'zh-TW' → 'ZhTW'. Mirrors fieldSuffix() in site/build.js, which
  // writes the translated fields into data.js under exactly these names.
  function fieldSuffix(lang) {
    return String(lang).split(/[^A-Za-z0-9]+/).filter(Boolean)
      .map(function (part) { return part.charAt(0).toUpperCase() + part.slice(1); })
      .join('');
  }

  // Kept for callers that switch language programmatically (tests, deep links).
  // The picker re-reads localStorage, so writing there keeps both in sync.
  function setLang(lang) {
    if (lang !== 'en' && !supported(lang)) return;
    try {
      if (lang === 'en') localStorage.removeItem(STORAGE_KEY);
      else localStorage.setItem(STORAGE_KEY, lang);
    } catch (e) { /* ignore */ }
    document.documentElement.setAttribute('lang', lang);
    window.dispatchEvent(new CustomEvent('aifs:langchange', { detail: { lang: lang } }));
  }

  function t(key) {
    var lang = getLang();
    return (UI[lang] && UI[lang][key]) || (UI.en && UI.en[key]) || key;
  }

  // pick(obj, 'name') → obj.nameKo under 'ko' when present, else obj.name
  function pick(obj, field) {
    if (!obj) return '';
    var lang = getLang();
    if (lang !== 'en') {
      var translated = obj[field + fieldSuffix(lang)];
      if (translated != null && translated !== '') return translated;
    }
    return obj[field] != null ? obj[field] : '';
  }

  // Apply i18n attributes on the current page:
  //   [data-i18n]             → textContent
  //   [data-i18n-placeholder] → placeholder attribute (inputs)
  //   [data-i18n-title]       → title attribute
  function applyLabels(root) {
    var scope = root || document;
    var text = scope.querySelectorAll('[data-i18n]');
    for (var i = 0; i < text.length; i++) {
      text[i].textContent = t(text[i].getAttribute('data-i18n'));
    }
    var ph = scope.querySelectorAll('[data-i18n-placeholder]');
    for (var p = 0; p < ph.length; p++) {
      ph[p].setAttribute('placeholder', t(ph[p].getAttribute('data-i18n-placeholder')));
    }
    var ti = scope.querySelectorAll('[data-i18n-title]');
    for (var k = 0; k < ti.length; k++) {
      ti[k].setAttribute('title', t(ti[k].getAttribute('data-i18n-title')));
    }
  }

  window.AIFSi18n = {
    getLang: getLang, setLang: setLang, t: t, pick: pick, applyLabels: applyLabels
  };

  // Re-apply labels whenever the picker (or setLang) changes the language.
  window.addEventListener('aifs:langchange', function () { applyLabels(document); });
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { applyLabels(document); });
  } else {
    applyLabels(document);
  }
})();
