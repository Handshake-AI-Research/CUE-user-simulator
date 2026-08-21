(() => {
  const state = {
    items: [],
    taxonomy: [],
    index: 0,
    decisions: {},
    batch: null,
    runDir: "",
  };

  const $ = (id) => document.getElementById(id);

  function esc(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function current() {
    return state.items[state.index];
  }

  function fillModes(selected) {
    const sel = $("mode-select");
    const names = state.taxonomy.map((m) => m.name);
    if (selected && !names.includes(selected)) names.push(selected);
    sel.innerHTML = names
      .map((n) => `<option value="${esc(n)}">${esc(n)}</option>`)
      .join("");
    if (selected) sel.value = selected;
  }

  function selectedModeNames() {
    return [...document.querySelectorAll(".taxonomy-item .pick:checked")].map(
      (el) => el.value
    );
  }

  function applyLabelMap(mapping) {
    for (const item of state.items) {
      if (item.proposal?.label && mapping[item.proposal.label]) {
        item.proposal.label = mapping[item.proposal.label];
      }
      const dec = state.decisions[item.primary_key];
      if (dec?.label && mapping[dec.label]) {
        dec.label = mapping[dec.label];
      }
      if (item.decision?.label && mapping[item.decision.label]) {
        item.decision.label = mapping[item.decision.label];
      }
    }
  }

  function renderTaxonomy() {
    const empty = state.taxonomy.filter((m) => !(m.n_samples > 0)).length;
    $("taxonomy-count").textContent =
      `${state.taxonomy.length} modes` + (empty ? ` · ${empty} empty` : "");
    const root = $("taxonomy-list");
    const checked = new Set(selectedModeNames());
    root.innerHTML = "";
    if (!state.taxonomy.length) {
      root.innerHTML = `<p class="muted">No modes yet — add one, or finish a propose batch first.</p>`;
      return;
    }
    // Show empty modes first so they're easy to spot / prune.
    const ordered = [...state.taxonomy].sort((a, b) => {
      const ae = a.n_samples > 0 ? 1 : 0;
      const be = b.n_samples > 0 ? 1 : 0;
      if (ae !== be) return ae - be;
      return String(a.name).localeCompare(String(b.name));
    });
    for (const mode of ordered) {
      const n = Number(mode.n_samples || 0);
      const row = document.createElement("div");
      row.className = "taxonomy-item" + (n === 0 ? " empty" : "");
      row.innerHTML = `
        <input type="checkbox" class="pick" value="${esc(mode.name)}"${
          checked.has(mode.name) ? " checked" : ""
        } />
        <span class="count-pill${n === 0 ? " zero" : ""}" title="proposal+decision+tagged samples">${n}</span>
        <input
          type="text"
          class="name-input"
          data-mode="${esc(mode.name)}"
          value="${esc(mode.name)}"
          placeholder="mode name"
        />
        <input
          type="text"
          class="desc-input"
          data-mode="${esc(mode.name)}"
          value="${esc(mode.description || "")}"
          placeholder="Add a description…"
        />
      `;
      root.appendChild(row);
    }
    for (const input of root.querySelectorAll(".desc-input")) {
      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") input.blur();
      });
      input.addEventListener("change", async () => {
        const name = input.dataset.mode;
        const mode = state.taxonomy.find((m) => m.name === name);
        if (!mode || input.value === (mode.description || "")) return;
        try {
          await saveDescription(name, input.value);
        } catch (err) {
          alert(`Save failed: ${err.message || err}`);
        }
      });
    }
    for (const input of root.querySelectorAll(".name-input")) {
      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") input.blur();
        if (ev.key === "Escape") {
          input.value = input.dataset.mode;
          input.blur();
        }
      });
      input.addEventListener("change", async () => {
        const old = input.dataset.mode;
        const neu = input.value.trim();
        if (!neu || neu === old) {
          input.value = old;
          return;
        }
        try {
          await renameModeInline(old, neu);
        } catch (err) {
          input.value = old;
          alert(`Rename failed: ${err.message || err}`);
        }
      });
    }
  }

  function renderChips(item) {
    const box = $("reward-chips");
    box.innerHTML = "";
    const failed = item.reward_report?.failed || [];
    for (const f of failed) {
      const chip = document.createElement("span");
      chip.className = "chip";
      const label =
        f.kind === "action"
          ? `action:${f.name || "?"}`
          : f.kind === "nl_assertion"
            ? `nl:${(f.nl_assertion || "").slice(0, 40)}`
            : f.kind === "communicate"
              ? `communicate:${(f.info || "").slice(0, 40)}`
              : String(f.kind || "fail");
      chip.textContent = label;
      box.appendChild(chip);
    }
  }

  function renderTranscript(item) {
    const cite = new Set(
      (state.decisions[item.primary_key]?.turn_indices ??
        item.proposal?.turn_indices ??
        []).map(Number)
    );
    const root = $("transcript");
    root.innerHTML = "";
    (item.conversation || []).forEach((turn, i) => {
      const el = document.createElement("article");
      el.className = "turn";
      if (cite.has(i)) el.classList.add("highlight");
      const role = String(turn.role || "").toLowerCase();
      if (role === "tool" || turn.tool_calls) el.classList.add("tool");
      let body = turn.content || "";
      if (turn.tool_calls?.length) {
        body =
          turn.tool_calls
            .map(
              (tc) =>
                `CALL ${tc.name}(${JSON.stringify(tc.arguments || {})}) id=${tc.id}`
            )
            .join("\n") + (body ? `\n${body}` : "");
      }
      if (role === "tool") {
        body = `RESULT id=${turn.tool_call_id || ""}${turn.error ? " ERROR" : ""}\n${body}`;
      }
      el.innerHTML = `<div class="turn-meta"><span>Turn ${i} · ${esc(role)}</span>${
        cite.has(i) ? "<span>cited</span>" : ""
      }</div><div class="turn-body">${esc(body)}</div>`;
      root.appendChild(el);
    });
  }

  function render() {
    const item = current();
    if (!item) return;
    $("meta").textContent = `${state.runDir} · batch ${state.batch}`;
    $("progress").textContent = `${state.index + 1} / ${state.items.length}`;
    renderTaxonomy();
    $("episode-header").innerHTML = `
      <div><strong>${esc(item.primary_key)}</strong></div>
      <div>${esc(item.domain)} / task ${esc(item.task_id)} · ${esc(item.source_id)}</div>
      <div>${esc(item.task_description || "")}</div>
      ${
        item.success_criteria
          ? `<details><summary>Success criteria</summary><pre>${esc(
              item.success_criteria
            )}</pre></details>`
          : ""
      }
    `;
    renderChips(item);
    renderTranscript(item);
    const prop = item.proposal || {};
    $("proposal").innerHTML = `
      <div><strong>${esc(prop.label || "(none)")}</strong></div>
      <div>${esc(prop.explanation || "")}</div>
      <div class="muted">Turns: ${(prop.turn_indices || []).join(", ") || "—"}</div>
      <div class="muted">${esc(prop.excerpt || "")}</div>
    `;
    const dec = state.decisions[item.primary_key];
    fillModes(dec?.label || prop.label || state.taxonomy[0]?.name || "");
    $("notes").value = dec?.notes || "";
    $("save-status").textContent = dec ? "Saved locally (finish batch to write)." : "";
  }

  function saveCurrent() {
    const item = current();
    if (!item) return;
    const prop = item.proposal || {};
    state.decisions[item.primary_key] = {
      primary_key: item.primary_key,
      label: $("mode-select").value,
      notes: $("notes").value,
      explanation: prop.explanation || "",
      turn_indices: prop.turn_indices || [],
      excerpt: prop.excerpt || "",
    };
    $("save-status").textContent = "Saved locally (finish batch to write).";
  }

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || data.message || res.statusText);
    }
    return data;
  }

  async function addMode(name, description = "") {
    const data = await postJson("/api/mode", { name, description });
    state.taxonomy = data.taxonomy || state.taxonomy;
    fillModes(name);
    renderTaxonomy();
    $("taxonomy-status").textContent = `Added “${name}”.`;
  }

  async function saveDescription(name, description) {
    const data = await postJson("/api/mode", { name, description });
    state.taxonomy = data.taxonomy || state.taxonomy;
    renderTaxonomy();
    $("taxonomy-status").textContent = description.trim()
      ? `Saved description for “${name}”.`
      : `Cleared description for “${name}”.`;
  }

  async function renameModeInline(old, neu) {
    const data = await postJson("/api/mode/rename", { old, new: neu });
    state.taxonomy = data.taxonomy || state.taxonomy;
    applyLabelMap({ [old]: neu });
    render();
    const r = data.remapped || {};
    $("taxonomy-status").textContent =
      `Renamed “${old}” → “${neu}” ` +
      `(rewrote proposals=${r.proposals || 0}, decisions=${r.decisions || 0}, tagged=${r.tagged || 0}).`;
  }

  $("btn-prev").onclick = () => {
    saveCurrent();
    state.index = Math.max(0, state.index - 1);
    render();
  };
  $("btn-next").onclick = () => {
    saveCurrent();
    state.index = Math.min(state.items.length - 1, state.index + 1);
    render();
  };
  $("btn-save").onclick = () => saveCurrent();

  async function promptNewMode() {
    const name = prompt("New failure mode name:");
    if (!name?.trim()) return;
    const description = prompt("Short description (optional):") || "";
    try {
      await addMode(name.trim(), description);
    } catch (err) {
      alert(`Add failed: ${err.message || err}`);
    }
  }

  $("btn-new-mode").onclick = promptNewMode;
  $("btn-add-mode").onclick = promptNewMode;

  $("btn-merge-modes").onclick = async () => {
    const selected = selectedModeNames();
    if (selected.length < 1) {
      alert("Select one or more source modes to merge.");
      return;
    }
    const names = state.taxonomy.map((m) => m.name);
    const target = prompt(
      `Merge [${selected.join(", ")}] into which mode?\n` +
        `Enter an existing name or a new name.\nAvailable: ${names.join(" · ")}`,
      selected[0]
    );
    if (!target?.trim()) return;
    const description =
      prompt("Description for the merged mode (optional):") || "";
    if (
      !confirm(
        `Merge ${selected.join(", ")} → “${target.trim()}”?\n` +
          "This rewrites labels in proposals/decisions/tagged for this run."
      )
    ) {
      return;
    }
    try {
      const data = await postJson("/api/mode/merge", {
        sources: selected,
        target: target.trim(),
        description,
      });
      state.taxonomy = data.taxonomy || state.taxonomy;
      const mapping = Object.fromEntries(selected.map((s) => [s, target.trim()]));
      applyLabelMap(mapping);
      render();
      const r = data.remapped || {};
      $("taxonomy-status").textContent =
        `Merged into “${target.trim()}” ` +
        `(rewrote proposals=${r.proposals || 0}, decisions=${r.decisions || 0}, tagged=${r.tagged || 0}).`;
    } catch (err) {
      alert(`Merge failed: ${err.message || err}`);
    }
  };

  $("btn-prune-empty").onclick = async () => {
    const empty = state.taxonomy.filter((m) => !(m.n_samples > 0)).map((m) => m.name);
    if (!empty.length) {
      $("taxonomy-status").textContent = "No empty modes to remove.";
      return;
    }
    if (
      !confirm(
        `Remove ${empty.length} mode(s) with 0 samples?\n\n` + empty.join("\n")
      )
    ) {
      return;
    }
    try {
      const data = await postJson("/api/mode/prune-empty", {});
      state.taxonomy = data.taxonomy || state.taxonomy;
      render();
      $("taxonomy-status").textContent =
        `Removed ${ (data.removed || []).length } empty mode(s).`;
    } catch (err) {
      alert(`Prune failed: ${err.message || err}`);
    }
  };

  $("btn-finish").onclick = async () => {
    saveCurrent();
    // Fill any undecided items with proposal labels.
    for (const item of state.items) {
      if (!state.decisions[item.primary_key]) {
        const prop = item.proposal || {};
        state.decisions[item.primary_key] = {
          primary_key: item.primary_key,
          label: prop.label || "Uncategorized",
          notes: "",
          explanation: prop.explanation || "",
          turn_indices: prop.turn_indices || [],
          excerpt: prop.excerpt || "",
        };
      }
    }
    if (!confirm(`Write ${Object.keys(state.decisions).length} decisions and update taxonomy?`)) {
      return;
    }
    try {
      const data = await postJson("/api/finish", {
        decisions: Object.values(state.decisions),
      });
      alert(`Wrote ${data.n_decisions} decisions.`);
    } catch (err) {
      alert(`Finish failed: ${err.message || err}`);
    }
  };

  document.addEventListener("keydown", (ev) => {
    if (ev.target.matches("textarea, input, select")) return;
    if (ev.key === "j") $("btn-next").click();
    if (ev.key === "k") $("btn-prev").click();
  });

  fetch("/api/state")
    .then((r) => r.json())
    .then((data) => {
      state.items = data.items || [];
      state.taxonomy = data.taxonomy || [];
      state.batch = data.batch;
      state.runDir = data.run_dir;
      for (const item of state.items) {
        if (item.decision) state.decisions[item.primary_key] = item.decision;
      }
      render();
    });
})();
