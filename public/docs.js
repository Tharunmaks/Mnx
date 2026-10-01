// Builds PDF, DOCX and PPTX files in the browser from the structured content
// Mnx sends through the create_document tool, and renders inline previews.
(() => {
  const THEMES = {
    midnight: { bg: "0F1221", fg: "F5F6FF", accent: "8B7CFF", muted: "A6ABC9" },
    ocean: { bg: "0B2540", fg: "F0F9FF", accent: "38BDF8", muted: "9CC3DD" },
    sunset: { bg: "2B1331", fg: "FFF5F0", accent: "FF8A5B", muted: "D9AFC0" },
    forest: { bg: "0F2A1F", fg: "F0FFF6", accent: "4ADE80", muted: "A3C9B4" },
    paper: { bg: "FFFFFF", fg: "1F2330", accent: "4F46E5", muted: "6B7280" },
  };

  const loaded = {};
  function loadScript(name) {
    loaded[name] ||= new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = `/vendor/${name}`;
      s.onload = resolve;
      s.onerror = () => reject(new Error(`Couldn't load ${name}`));
      document.head.appendChild(s);
    });
    return loaded[name];
  }

  // Parse the simple markdown we allow in bodies into typed lines.
  function parseBody(body) {
    const out = [];
    for (const raw of String(body || "").split(/\r?\n/)) {
      const line = raw.trimEnd();
      if (!line.trim()) {
        out.push({ type: "gap" });
        continue;
      }
      let m;
      if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) out.push({ type: "bullet", text: m[1] });
      else if ((m = line.match(/^\s*(\d+)[.)]\s+(.*)$/))) out.push({ type: "number", n: m[1], text: m[2] });
      else if ((m = line.match(/^\s*#{1,6}\s+(.*)$/))) out.push({ type: "sub", text: m[1] });
      else out.push({ type: "para", text: line.trim() });
    }
    return out;
  }
  const runs = (text) =>
    String(text)
      .split(/(\*\*[^*]+\*\*)/g)
      .filter(Boolean)
      .map((t) => (t.startsWith("**") && t.endsWith("**") ? { text: t.slice(2, -2), bold: true } : { text: t.replace(/\*/g, ""), bold: false }));
  const plain = (text) => String(text).replace(/\*\*/g, "").replace(/`/g, "");

  async function buildPptx(doc) {
    await loadScript("pptxgen.js");
    const t = THEMES[doc.theme] || THEMES.midnight;
    const pptx = new window.PptxGenJS();
    pptx.layout = "LAYOUT_WIDE";
    pptx.title = doc.title;
    const face = "Calibri";

    const cover = pptx.addSlide();
    cover.background = { color: t.bg };
    cover.addShape("rect", { x: 0.7, y: 2.2, w: 1.4, h: 0.09, fill: { color: t.accent }, line: { color: t.accent } });
    cover.addText(doc.title, { x: 0.7, y: 2.4, w: 11.9, h: 1.5, fontFace: face, fontSize: 44, bold: true, color: t.fg, valign: "top" });
    if (doc.subtitle) cover.addText(doc.subtitle, { x: 0.7, y: 3.9, w: 11.9, h: 0.9, fontFace: face, fontSize: 20, color: t.muted, valign: "top" });

    doc.slides.forEach((s, idx) => {
      const slide = pptx.addSlide();
      slide.background = { color: t.bg };
      slide.addShape("rect", { x: 0.7, y: 0, w: 1.4, h: 0.09, fill: { color: t.accent }, line: { color: t.accent } });
      slide.addText(plain(s.title), { x: 0.7, y: 0.45, w: 11.9, h: 1.0, fontFace: face, fontSize: 32, bold: true, color: t.fg, valign: "middle" });
      if (s.bullets.length) {
        slide.addText(
          s.bullets.map((b) => ({ text: plain(b), options: { bullet: { indent: 22 }, breakLine: true, paraSpaceAfter: 10 } })),
          { x: 0.8, y: 1.65, w: 11.7, h: 5.2, fontFace: face, fontSize: s.bullets.length > 6 ? 17 : 20, color: t.fg, valign: "top" },
        );
      }
      slide.addText(String(idx + 2), { x: 12.2, y: 6.9, w: 0.8, h: 0.4, fontSize: 11, color: t.muted, align: "right" });
      if (s.notes) slide.addNotes(s.notes);
    });
    return pptx.write({ outputType: "blob" });
  }

  async function buildPdf(doc) {
    await loadScript("jspdf.js");
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({ unit: "pt", format: "a4" });
    const t = THEMES[doc.theme] || THEMES.midnight;
    const hex = (h) => [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
    const W = pdf.internal.pageSize.getWidth();
    const H = pdf.internal.pageSize.getHeight();
    const M = 56;
    const maxW = W - M * 2;
    let y = M;

    const ensure = (h) => {
      if (y + h > H - M) {
        pdf.addPage();
        y = M;
      }
    };
    const write = (text, { size = 11, style = "normal", color = [40, 44, 60], indent = 0, gap = 4, prefix = "" } = {}) => {
      pdf.setFont("helvetica", style);
      pdf.setFontSize(size);
      pdf.setTextColor(...color);
      const lines = pdf.splitTextToSize(plain(text), maxW - indent - (prefix ? 14 : 0));
      const lh = size * 1.45;
      lines.forEach((ln, i) => {
        ensure(lh);
        if (prefix && i === 0) pdf.text(prefix, M + indent, y + size);
        pdf.text(ln, M + indent + (prefix ? 14 : 0), y + size);
        y += lh;
      });
      y += gap;
    };

    const accent = t.bg === "FFFFFF" ? hex(t.accent) : hex(t.accent);
    pdf.setFillColor(...accent);
    pdf.rect(M, y, 60, 4, "F");
    y += 18;
    write(doc.title, { size: 24, style: "bold", color: [17, 17, 34], gap: 6 });
    if (doc.subtitle) write(doc.subtitle, { size: 13, color: [110, 114, 135], gap: 14 });

    for (const s of doc.sections) {
      if (s.heading) {
        y += 8;
        ensure(40);
        write(s.heading, { size: 15, style: "bold", color: accent, gap: 4 });
      }
      for (const line of parseBody(s.body)) {
        if (line.type === "gap") y += 5;
        else if (line.type === "bullet") write(line.text, { prefix: "•", indent: 6, gap: 2 });
        else if (line.type === "number") write(line.text, { prefix: `${line.n}.`, indent: 4, gap: 2 });
        else if (line.type === "sub") write(line.text, { size: 12.5, style: "bold", gap: 2 });
        else write(line.text, { gap: 5 });
      }
    }
    const pages = pdf.getNumberOfPages();
    for (let p = 1; p <= pages; p++) {
      pdf.setPage(p);
      pdf.setFontSize(9);
      pdf.setTextColor(150);
      pdf.text(`${p} / ${pages}`, W - M, H - 28, { align: "right" });
    }
    return pdf.output("blob");
  }

  async function buildDocx(doc) {
    await loadScript("docx.js");
    const { Document, Packer, Paragraph, TextRun, HeadingLevel, LevelFormat, AlignmentType } = window.docx;
    const children = [new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(doc.title)] })];
    if (doc.subtitle) children.push(new Paragraph({ children: [new TextRun({ text: doc.subtitle, italics: true, color: "666666" })] }));
    for (const s of doc.sections) {
      if (s.heading) children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(plain(s.heading))] }));
      let numbered = 0;
      for (const line of parseBody(s.body)) {
        if (line.type === "gap") continue;
        const kids = runs(line.text).map((r) => new TextRun({ text: r.text, bold: r.bold }));
        if (line.type === "bullet") children.push(new Paragraph({ bullet: { level: 0 }, children: kids }));
        else if (line.type === "number") {
          numbered++;
          children.push(new Paragraph({ numbering: { reference: "num", level: 0 }, children: kids }));
        } else if (line.type === "sub") children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: kids }));
        else children.push(new Paragraph({ children: kids, spacing: { after: 120 } }));
      }
      void numbered;
    }
    const document_ = new Document({
      creator: "Mnx",
      title: doc.title,
      numbering: {
        config: [
          {
            reference: "num",
            levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START }],
          },
        ],
      },
      sections: [{ children }],
    });
    return Packer.toBlob(document_);
  }

  async function build(doc) {
    if (doc.format === "pptx") return buildPptx(doc);
    if (doc.format === "pdf") return buildPdf(doc);
    return buildDocx(doc);
  }

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const inline = (s) => esc(s).replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");

  // Inline preview element for a document card.
  function preview(doc) {
    const wrap = document.createElement("div");
    if (doc.format === "pptx") {
      const t = THEMES[doc.theme] || THEMES.midnight;
      const deck = [{ cover: true, title: doc.title, sub: doc.subtitle }, ...doc.slides];
      let i = 0;
      wrap.className = "slides-view";
      const stage = document.createElement("div");
      const nav = document.createElement("div");
      nav.className = "slide-nav";
      nav.innerHTML = `<button class="icon-btn" data-d="-1" aria-label="Previous slide">‹</button><span></span><button class="icon-btn" data-d="1" aria-label="Next slide">›</button>`;
      const show = () => {
        const s = deck[i];
        stage.innerHTML = `<div class="slide ${s.cover ? "cover" : ""}" style="background:#${t.bg};color:#${t.fg};--s-accent:#${t.accent}">
          <h3>${inline(s.title)}</h3>${s.cover ? (s.sub ? `<p>${inline(s.sub)}</p>` : "") : `<ul>${(s.bullets || []).map((b) => `<li>${inline(b)}</li>`).join("")}</ul>`}</div>`;
        nav.querySelector("span").textContent = `${i + 1} / ${deck.length}`;
      };
      nav.addEventListener("click", (e) => {
        const d = e.target.closest("[data-d]");
        if (!d) return;
        i = (i + Number(d.dataset.d) + deck.length) % deck.length;
        show();
      });
      wrap.append(stage, nav);
      show();
    } else {
      wrap.className = "page-view";
      let html = `<h3>${inline(doc.title)}</h3>${doc.subtitle ? `<div class="sub">${inline(doc.subtitle)}</div>` : ""}`;
      for (const s of doc.sections) {
        if (s.heading) html += `<h4>${inline(s.heading)}</h4>`;
        let list = null;
        for (const line of parseBody(s.body)) {
          const want = line.type === "bullet" ? "ul" : line.type === "number" ? "ol" : null;
          if (list && list !== want) {
            html += `</${list}>`;
            list = null;
          }
          if (want && !list) {
            html += `<${want}>`;
            list = want;
          }
          if (want) html += `<li>${inline(line.text)}</li>`;
          else if (line.type === "sub") html += `<p><b>${inline(line.text)}</b></p>`;
          else if (line.type === "para") html += `<p>${inline(line.text)}</p>`;
        }
        if (list) html += `</${list}>`;
      }
      wrap.innerHTML = html;
    }
    return wrap;
  }

  window.MnxDocs = { build, preview };
})();
