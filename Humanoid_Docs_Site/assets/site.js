/* Humanoid Docs — shared nav + rich interactions. No deps; works on file://. */
(function () {
  const CHAPTERS = [
    { sec: "Start" },
    { id: "index",           file: "index.html",                    ix: "00", icon: "🏠", title: "Home" },
    { sec: "Foundations" },
    { id: "big_picture",     file: "chapters/01_big_picture.html",  ix: "01", icon: "🌍", title: "The Big Picture" },
    { id: "the_stack",       file: "chapters/02_the_stack.html",    ix: "02", icon: "🧱", title: "The Tech Stack" },
    { id: "sit_to_stand",    file: "chapters/03_sit_to_stand.html", ix: "03", icon: "🧍", title: "Sit → Stand" },
    { sec: "How It Works" },
    { id: "rewards",         file: "chapters/04_rewards.html",      ix: "04", icon: "🎯", title: "Reward Design" },
    { id: "simulation",      file: "chapters/05_simulation.html",   ix: "05", icon: "🌀", title: "Simulation Setup" },
    { id: "results",         file: "chapters/06_results.html",      ix: "06", icon: "📊", title: "Evaluation & Results" },
    { id: "language_vision", file: "chapters/07_language_vision.html",ix:"07",icon: "🗣️", title: "Language & Vision" },
    { sec: "Projects & Choices" },
    { id: "wakeboarding",    file: "chapters/08_wakeboarding.html", ix: "08", icon: "🏄", title: "Wakeboarding RL" },
    { id: "model_choices",   file: "chapters/09_model_choices.html",ix: "09", icon: "🧠", title: "Why These Choices" },
    { id: "compute",         file: "chapters/10_compute.html",      ix: "10", icon: "☁️", title: "Compute & Workflow" },
    { sec: "Wrap-up" },
    { id: "roadmap",         file: "chapters/11_roadmap.html",      ix: "11", icon: "🗺️", title: "Roadmap" },
    { id: "glossary",        file: "chapters/12_glossary.html",     ix: "12", icon: "📖", title: "Glossary" },
  ];

  const HSite = {
    init(activeId) {
      this.active = activeId;
      const inCh = location.pathname.replace(/\\/g, "/").includes("/chapters/");
      this.root = inCh ? "../" : "./";
      this.chrome(); this.buildNav(); this.search(); this.counters();
      this.toc(); this.anchors(); this.reveal(); this.keys();
    },

    chrome() {
      const bar = el("div", { id: "progress" }); document.body.appendChild(bar);
      const btn = el("button", { class: "menu-btn", html: "☰" });
      btn.onclick = () => document.getElementById("sidebar").classList.toggle("open");
      document.body.appendChild(btn);
      const top = el("button", { id: "toTop", html: "↑", title: "Back to top" });
      top.onclick = () => window.scrollTo({ top: 0, behavior: "smooth" });
      document.body.appendChild(top);
      document.addEventListener("click", (e) => {
        const sb = document.getElementById("sidebar");
        if (window.innerWidth <= 960 && sb.classList.contains("open") && !sb.contains(e.target) && !btn.contains(e.target)) sb.classList.remove("open");
      });
      const onScroll = () => {
        const s = document.documentElement.scrollTop;
        const h = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        bar.style.width = (h > 0 ? (s / h) * 100 : 0) + "%";
        top.classList.toggle("show", s > 500);
        this.spy();
      };
      document.addEventListener("scroll", onScroll, { passive: true }); onScroll();
    },

    buildNav() {
      const sb = document.getElementById("sidebar");
      let nav = "";
      CHAPTERS.forEach((c) => {
        if (c.sec) { nav += `<div class="nav-sec">${c.sec}</div>`; return; }
        const cls = c.id === this.active ? "active" : "";
        nav += `<a class="${cls}" data-t="${c.title.toLowerCase()}" href="${this.root}${c.file}"><span class="ix">${c.ix}</span><span class="ico">${c.icon}</span><span>${c.title}</span></a>`;
      });
      sb.innerHTML =
        `<a class="brand" href="${this.root}index.html"><span class="logo">🤖</span><span><b>Humanoid RL</b><small>Interactive Docs</small></span></a>
         <div class="search"><input id="q" type="text" placeholder="Search chapters…" autocomplete="off"><kbd>/</kbd></div>
         <div class="nav" id="nav">${nav}</div>`;
    },

    search() {
      const q = document.getElementById("q"), nav = document.getElementById("nav");
      const run = () => {
        const v = q.value.trim().toLowerCase();
        let shown = 0;
        nav.querySelectorAll("a").forEach(a => {
          const ok = !v || a.dataset.t.includes(v);
          a.classList.toggle("hidden", !ok); if (ok) shown++;
        });
        nav.querySelectorAll(".nav-sec").forEach(s => s.style.display = v ? "none" : "");
        let e = nav.querySelector(".empty"); if (e) e.remove();
        if (!shown) nav.insertAdjacentHTML("beforeend", `<div class="empty">No matches</div>`);
      };
      q.addEventListener("input", run);
      document.addEventListener("keydown", (e) => {
        if (e.key === "/" && document.activeElement !== q) { e.preventDefault(); q.focus(); }
        if (e.key === "Escape") { q.value = ""; run(); q.blur(); }
      });
    },

    counters() {
      document.querySelectorAll(".stat .n").forEach(node => {
        const raw = node.textContent;
        const m = raw.match(/([\d,]+\.?\d*)/);
        if (!m) return;
        const target = parseFloat(m[1].replace(/,/g, "")); if (!isFinite(target) || target < 1) return;
        const dec = (m[1].split(".")[1] || "").length;
        const pre = raw.slice(0, m.index), suf = raw.slice(m.index + m[1].length);
        const original = node.innerHTML;        // restore exact markup when done
        let started = false;
        const animate = () => {
          if (started) return; started = true;
          const dur = 1100, t0 = performance.now();
          const tick = (t) => {
            const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
            node.textContent = pre + (target * e).toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suf;
            if (k < 1) requestAnimationFrame(tick);
            else node.innerHTML = original;     // exact original (keeps nested spans/styling)
          };
          requestAnimationFrame(tick);
        };
        if ("IntersectionObserver" in window) {
          const io = new IntersectionObserver(es => es.forEach(x => { if (x.isIntersecting) { animate(); io.disconnect(); } }), { threshold: .4 });
          io.observe(node.closest(".stat"));
        } else animate();
      });
    },

    anchors() {
      document.querySelectorAll("main h2").forEach((h, i) => {
        if (!h.id) h.id = "s" + i;
        const a = el("a", { class: "anchor", html: "#" }); a.href = "#" + h.id;
        h.appendChild(a);
      });
    },

    toc() {
      const hs = [...document.querySelectorAll("main h2")];
      if (hs.length < 2) return;
      const box = el("nav", { id: "toc" });
      box.innerHTML = `<div class="h">On this page</div>` +
        hs.map((h, i) => { if (!h.id) h.id = "s" + i; return `<a href="#${h.id}">${h.firstChild.textContent.trim()}</a>`; }).join("");
      document.body.appendChild(box);
      this._toc = box; this._hs = hs;
    },
    spy() {
      if (!this._toc) return;
      let cur = this._hs[0];
      for (const h of this._hs) { if (h.getBoundingClientRect().top <= 130) cur = h; }
      this._toc.querySelectorAll("a").forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + cur.id));
    },

    keys() {
      const items = CHAPTERS.filter(c => !c.sec);
      const i = items.findIndex(c => c.id === this.active);
      document.addEventListener("keydown", (e) => {
        if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
        if (e.key === "ArrowRight" && items[i + 1]) location.href = this.root + items[i + 1].file;
        if (e.key === "ArrowLeft" && items[i - 1]) location.href = this.root + items[i - 1].file;
      });
    },

    reveal() {
      const els = document.querySelectorAll(".rv");
      if (!("IntersectionObserver" in window)) { els.forEach(e => e.classList.add("in")); return; }
      const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: .12 });
      els.forEach(e => io.observe(e));
    },

    pager(root, activeId) {
      const items = CHAPTERS.filter(c => !c.sec);
      const i = items.findIndex(c => c.id === activeId);
      const p = items[i - 1], n = items[i + 1];
      return `<div class="pager">${
        p ? `<a href="${root}${p.file}"><span class="dir">← Previous</span><span class="t">${p.icon} ${p.title}</span></a>` : `<span></span>`}${
        n ? `<a class="next" href="${root}${n.file}"><span class="dir">Next →  (or press →)</span><span class="t">${n.icon} ${n.title}</span></a>` : `<span></span>`}</div>`;
    }
  };

  function el(tag, o) { const e = document.createElement(tag); if (o) { if (o.id) e.id = o.id; if (o.class) e.className = o.class; if (o.html != null) e.innerHTML = o.html; if (o.title) e.title = o.title; } return e; }
  window.HSite = HSite;
})();
