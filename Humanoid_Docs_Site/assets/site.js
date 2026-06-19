/* Humanoid Docs — shared nav + interactions. No external deps; works on file://. */
(function () {
  const CHAPTERS = [
    { sec: "Start" },
    { id: "index",            file: "index.html",                  ix: "00", icon: "🏠", title: "Home" },
    { sec: "Foundations" },
    { id: "big_picture",      file: "chapters/01_big_picture.html",ix: "01", icon: "🌍", title: "The Big Picture" },
    { id: "the_stack",        file: "chapters/02_the_stack.html",  ix: "02", icon: "🧱", title: "The Tech Stack" },
    { id: "sit_to_stand",     file: "chapters/03_sit_to_stand.html",ix:"03", icon: "🧍", title: "Sit → Stand" },
    { sec: "How It Works" },
    { id: "rewards",          file: "chapters/04_rewards.html",    ix: "04", icon: "🎯", title: "Reward Design" },
    { id: "simulation",       file: "chapters/05_simulation.html", ix: "05", icon: "🌀", title: "Simulation Setup" },
    { id: "results",          file: "chapters/06_results.html",    ix: "06", icon: "📊", title: "Evaluation & Results" },
    { id: "language_vision",  file: "chapters/07_language_vision.html",ix:"07",icon:"🗣️", title: "Language & Vision" },
    { sec: "Projects & Choices" },
    { id: "wakeboarding",     file: "chapters/08_wakeboarding.html",ix:"08", icon: "🏄", title: "Wakeboarding RL" },
    { id: "model_choices",    file: "chapters/09_model_choices.html",ix:"09",icon:"🧠", title: "Why These Choices" },
    { id: "compute",          file: "chapters/10_compute.html",    ix: "10", icon: "☁️", title: "Compute & Workflow" },
    { sec: "Wrap-up" },
    { id: "roadmap",          file: "chapters/11_roadmap.html",    ix: "11", icon: "🗺️", title: "Roadmap" },
    { id: "glossary",         file: "chapters/12_glossary.html",   ix: "12", icon: "📖", title: "Glossary" },
  ];

  const HSite = {
    init(activeId) {
      const inChapters = location.pathname.replace(/\\/g, "/").includes("/chapters/");
      const root = inChapters ? "../" : "./";
      this.buildChrome(root);
      this.buildNav(root, activeId);
      this.progress();
      this.reveal();
    },
    buildChrome(root) {
      const bar = document.createElement("div"); bar.id = "progress"; document.body.appendChild(bar);
      const btn = document.createElement("button"); btn.className = "menu-btn"; btn.innerHTML = "☰";
      btn.onclick = () => document.getElementById("sidebar").classList.toggle("open");
      document.body.appendChild(btn);
      document.addEventListener("click", (e) => {
        const sb = document.getElementById("sidebar");
        if (window.innerWidth <= 920 && sb.classList.contains("open") &&
            !sb.contains(e.target) && !btn.contains(e.target)) sb.classList.remove("open");
      });
    },
    buildNav(root, activeId) {
      const sb = document.getElementById("sidebar");
      let h = `<a class="brand" href="${root}index.html" style="text-decoration:none;color:inherit">
        <span class="logo">🤖</span><span><b>Humanoid RL</b><span>Interactive Docs</span></span></a>`;
      let nav = "";
      CHAPTERS.forEach((c) => {
        if (c.sec) { nav += `<div class="nav-sec">${c.sec}</div>`; return; }
        const cls = c.id === activeId ? "active" : "";
        nav += `<a class="${cls}" href="${root}${c.file}"><span class="ix">${c.ix}</span><span>${c.icon}</span><span>${c.title}</span></a>`;
      });
      sb.innerHTML = h + `<div class="nav">${nav}</div>`;
    },
    progress() {
      const bar = document.getElementById("progress");
      const upd = () => {
        const s = document.documentElement.scrollTop;
        const h = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        bar.style.width = (h > 0 ? (s / h) * 100 : 0) + "%";
      };
      document.addEventListener("scroll", upd, { passive: true }); upd();
    },
    reveal() {
      const els = document.querySelectorAll(".rv");
      if (!("IntersectionObserver" in window)) { els.forEach(e => e.classList.add("in")); return; }
      const io = new IntersectionObserver((es) => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: .12 });
      els.forEach(e => io.observe(e));
    },
    // builds prev/next pager from the ordered chapter list
    pager(root, activeId) {
      const items = CHAPTERS.filter(c => !c.sec);
      const i = items.findIndex(c => c.id === activeId);
      const prev = items[i - 1], next = items[i + 1];
      let h = "";
      h += prev ? `<a href="${root}${prev.file}"><span class="dir">← Previous</span><span class="t">${prev.icon} ${prev.title}</span></a>` : `<span></span>`;
      h += next ? `<a class="next" href="${root}${next.file}"><span class="dir">Next →</span><span class="t">${next.icon} ${next.title}</span></a>` : `<span></span>`;
      return `<div class="pager">${h}</div>`;
    }
  };
  window.HSite = HSite;
})();
