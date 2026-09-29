function initializeTailwind() {
      document.documentElement.style.setProperty('--accent', '#3b82f6');
    }

    function switchView(view) {
      document.querySelectorAll('.view').forEach(el => el.classList.add('hidden'));
      document.getElementById('view-' + view).classList.remove('hidden');

      // Update nav active state
      document.querySelectorAll('[id^="nav-"]').forEach(el => el.classList.remove('nav-active'));
      const navEl = document.getElementById('nav-' + view);
      if (navEl) navEl.classList.add('nav-active');

      if (view === 'writer') {
        loadWriterOptions();
        loadWriterHistory();
      }
      if (view === 'ingest') {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => initNeuralIngestor());
        });
      }
      if (view === 'learn') {
        loadLearnHistory();
      }
    }

    function showToast(msg, type = 'success') {
      const container = document.getElementById('toast-container');
      const colors = { success: 'bg-emerald-600', error: 'bg-red-600', info: 'bg-zinc-700' };
      const el = document.createElement('div');
      el.className = `toast px-5 py-3 rounded-2xl shadow-xl flex items-center gap-3 text-sm ${colors[type] || colors.info}`;
      el.innerHTML = `<span>${msg}</span>`;
      container.appendChild(el);
      setTimeout(() => { el.style.transition = 'all 0.2s'; el.style.opacity = '0'; setTimeout(() => el.remove(), 200); }, 2800);
    }


    // ========== DYNAMIC NAV FROM REGISTRY ==========
    async function loadNav() {
      try {
        const res = await fetch('/api/ui/nav');
        const data = await res.json();
        const navContainer = document.getElementById('nav-buttons');
        if (!navContainer) return;
        navContainer.innerHTML = data.items.map(item => `
          <button data-view="${item.key}" id="nav-${item.key}"
                  class="px-5 py-2 rounded-xl flex items-center gap-2 transition-all ${item.key === 'chat' ? 'nav-active' : ''}"
                  aria-label="${escapeHtml(item.label)}">
            <i class="fa-solid ${item.icon || 'fa-circle'}"></i><span>${escapeHtml(item.label)}</span>
          </button>
        `).join('');
        // Re-bind click handlers
        navContainer.querySelectorAll('[data-view]').forEach(btn => {
          btn.addEventListener('click', () => switchView(btn.dataset.view));
        });
      } catch (e) {
        console.warn('Failed to load nav from registry:', e);
      }
    }

    // ========== COMMAND PALETTE ==========
    let commandPaletteOpen = false;
    let commandPaletteItems = [];

    async function loadCommands() {
      try {
        const res = await fetch('/api/ui/commands');
        const data = await res.json();
        commandPaletteItems = data.items;
      } catch (e) {
        console.warn('Failed to load commands from registry:', e);
      }
    }

    function openCommandPalette() {
      if (commandPaletteOpen) return;
      commandPaletteOpen = true;
      const container = document.createElement('div');
      container.id = 'command-palette-overlay';
      container.className = 'fixed inset-0 z-[200] flex items-start justify-center pt-20';
      container.innerHTML = `
        <div class="bg-zinc-950/95 backdrop-blur border border-zinc-800 rounded-3xl shadow-2xl w-full max-w-2xl overflow-hidden animate-slideDown">
          <div class="flex items-center gap-3 p-4 border-b border-zinc-800">
            <i class="fa-solid fa-terminal text-cyan-400 text-xl"></i>
            <input type="text" id="command-palette-input" placeholder="Gõ lệnh hoặc tìm kiếm..." class="flex-1 bg-transparent border-none outline-none text-lg text-zinc-100 font-mono" autocomplete="off" spellcheck="false">
            <kbd class="px-3 py-1 text-[10px] bg-zinc-800 rounded text-zinc-500">Esc</kbd>
          </div>
          <div id="command-palette-list" class="max-h-96 overflow-y-auto"></div>
        </div>`;
      document.body.appendChild(container);
      const input = document.getElementById('command-palette-input');
      input.focus();
      renderCommandPalette('');
      input.addEventListener('input', (e) => renderCommandPalette(e.target.value));
      document.addEventListener('keydown', handleCommandPaletteKeydown);
      container.addEventListener('click', (e) => { if (e.target === container) closeCommandPalette(); });
    }

    function closeCommandPalette() {
      commandPaletteOpen = false;
      const overlay = document.getElementById('command-palette-overlay');
      if (overlay) overlay.remove();
      document.removeEventListener('keydown', handleCommandPaletteKeydown);
    }

    // Registry actions are data, not code. Only these names may run, and the
    // only accepted argument shape is a single quoted identifier.
    const COMMAND_ACTIONS = {
      switchView: (view) => switchView(view),
      newConversation: () => newConversation(),
      loadConversations: () => loadConversations(),
      loadMemoryGraph: () => loadMemoryGraph(),
      syncVault: () => syncVault(),
      showStats: () => showStats(),
    };

    function runCommandAction(action) {
      const match = /^([A-Za-z_$][\w$]*)\(\s*(?:'([\w-]*)'|"([\w-]*)")?\s*\)$/.exec(action || '');
      if (!match) {
        console.warn('Rejected registry action:', action);
        return;
      }
      const fn = COMMAND_ACTIONS[match[1]];
      if (!fn) {
        console.warn('Unknown registry action:', match[1]);
        return;
      }
      fn(match[2] ?? match[3]);
    }

    function handleCommandPaletteKeydown(e) {
      if (!commandPaletteOpen) return;
      if (e.key === 'Escape') {
        closeCommandPalette();
        return;
      }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const list = document.getElementById('command-palette-list');
        const items = list.querySelectorAll('.cmd-item');
        const current = list.querySelector('.cmd-item.selected');
        let idx = current ? Array.from(items).indexOf(current) : -1;
        idx += e.key === 'ArrowDown' ? 1 : -1;
        idx = Math.max(0, Math.min(items.length - 1, idx));
        items.forEach((it, i) => it.classList.toggle('selected', i === idx));
        items[idx]?.scrollIntoView({ block: 'nearest' });
      }
      if (e.key === 'Enter') {
        const selected = document.querySelector('#command-palette-list .cmd-item.selected');
        if (selected && selected.dataset.action) {
          runCommandAction(selected.dataset.action);
          closeCommandPalette();
        }
      }
    }

    function renderCommandPalette(query) {
      const list = document.getElementById('command-palette-list');
      if (!list) return;
      const q = (query || '').toLowerCase().trim();
      const filtered = commandPaletteItems
        .filter(item => !q || item.label.toLowerCase().includes(q) || (item.action || '').toLowerCase().includes(q))
        .map(item => `
          <button class="cmd-item w-full px-4 py-3 text-left hover:bg-zinc-900 border-b border-zinc-800/50 last:border-0 flex items-center gap-3 transition-colors ${q ? '' : 'selected'}"
                  data-action="${escapeHtml(item.action || '')}"
                  data-key="${escapeHtml(item.key)}">
            <i class="fa-solid ${item.icon || 'fa-circle'} text-cyan-400 text-lg"></i>
            <span class="text-zinc-100">${escapeHtml(item.label)}</span>
            ${item.group_name ? `<span class="ml-auto text-[10px] text-zinc-500 uppercase">${escapeHtml(item.group_name)}</span>` : ''}
          </button>
        `).join('');
      list.innerHTML = filtered || '<div class="px-4 py-6 text-center text-zinc-500">Không tìm thấy lệnh phù hợp</div>';
      list.querySelectorAll('.cmd-item').forEach(btn => {
        btn.addEventListener('click', () => {
          runCommandAction(btn.dataset.action);
          closeCommandPalette();
        });
      });
    }

    // ========== CHAT ==========
    let chatHistory = [];
    let currentConversationId = null;

    function newConversation() {
      currentConversationId = null;
      const win = document.getElementById('chat-window');
      if (win) win.innerHTML = '';
      const label = document.getElementById('chat-conv-label');
      if (label) label.textContent = 'Cuộc trò chuyện mới';
      const panel = document.getElementById('chat-history-panel');
      if (panel) panel.classList.add('hidden');
    }

    async function loadConversations() {
      const panel = document.getElementById('chat-history-panel');
      if (!panel) return;
      if (!panel.classList.contains('hidden') && panel.dataset.loaded === '1') {
        panel.classList.add('hidden');
        return;
      }
      panel.classList.remove('hidden');
      panel.innerHTML = '<div class="text-zinc-500 px-2 py-1">Đang tải lịch sử...</div>';
      try {
        const res = await fetch('/api/conversations?limit=20');
        const data = await res.json();
        const items = data.conversations || [];
        if (!items.length) {
          panel.innerHTML = '<div class="text-zinc-500 px-2 py-1">Chưa có cuộc trò chuyện nào.</div>';
          panel.dataset.loaded = '1';
          return;
        }
        panel.innerHTML = items.map(c => `
          <button onclick="openConversation('${escapeHtml(c.id)}')"
                  class="w-full text-left px-3 py-2 rounded-xl hover:bg-zinc-900 border border-transparent hover:border-zinc-800 transition-all">
            <div class="text-zinc-200 truncate">${escapeHtml(c.title || 'Chat')}</div>
            <div class="text-[10px] text-zinc-500 truncate mt-0.5">${escapeHtml(c.preview || '')}</div>
          </button>`).join('');
        panel.dataset.loaded = '1';
      } catch (e) {
        panel.innerHTML = `<div class="text-red-400 px-2 py-1">Lỗi: ${escapeHtml(e.message)}</div>`;
      }
    }

    async function openConversation(id) {
      currentConversationId = id;
      const panel = document.getElementById('chat-history-panel');
      if (panel) panel.classList.add('hidden');
      const win = document.getElementById('chat-window');
      if (!win) return;
      win.innerHTML = '<div class="text-zinc-500 text-sm">Đang tải tin nhắn...</div>';
      try {
        const res = await fetch(`/api/conversations/${encodeURIComponent(id)}/messages`);
        const data = await res.json();
        win.innerHTML = '';
        const label = document.getElementById('chat-conv-label');
        if (label) label.textContent = 'Đang tiếp tục · ' + id.slice(0, 8);
        (data.messages || []).forEach(m => {
          if (m.role === 'user') {
            win.innerHTML += `<div class="flex justify-end"><div class="chat-bubble-user px-5 py-3 max-w-[78%] text-sm">${escapeHtml(m.content || '')}</div></div>`;
          } else if (m.role === 'assistant') {
            win.innerHTML += renderAssistantBubble(m.content || '', null, m.citations || []);
          }
        });
        win.scrollTop = win.scrollHeight;
      } catch (e) {
        win.innerHTML = `<div class="text-red-400 text-xs">Lỗi: ${escapeHtml(e.message)}</div>`;
      }
    }

    function renderCitations(citations) {
      if (!citations || !citations.length) return '';
      const pills = citations
        .filter(c => c && c.url)
        .slice(0, 12)
        .map(c => {
          const title = escapeHtml((c.title || c.url || '').slice(0, 60));
          const url = escapeHtml(c.url);
          return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-pill" title="${url}"><i class="fa-solid fa-link text-[9px] opacity-70"></i><span class="truncate">${title}</span></a>`;
        }).join('');
      if (!pills) return '';
      return `<div class="mt-2 flex flex-wrap items-center gap-0.5"><span class="text-[10px] text-zinc-500 mr-1">Nguồn:</span>${pills}</div>`;
    }

    function renderAssistantBubble(content, contentHtml, citations) {
      const body = contentHtml
        ? `<div class="md-body">${contentHtml}</div>`
        : escapeHtml(content || '');
      return `<div class="flex"><div class="chat-bubble-ai px-5 py-4 max-w-[82%] text-sm">${body}${renderCitations(citations)}</div></div>`;
    }

    async function sendMessage() {
      const input = document.getElementById('chat-input');
      const msg = input.value.trim();
      if (!msg) return;

      const chatWin = document.getElementById('chat-window');
      chatWin.innerHTML += `<div class="flex justify-end"><div class="chat-bubble-user px-5 py-3 max-w-[78%] text-sm">${escapeHtml(msg)}</div></div>`;
      chatWin.scrollTop = chatWin.scrollHeight;
      input.value = '';

      // Show typing
      const typingId = 'typing-' + Date.now();
      chatWin.innerHTML += `<div id="${typingId}" class="flex"><div class="chat-bubble-ai px-5 py-3 text-sm text-zinc-400">Đang suy nghĩ...</div></div>`;
      chatWin.scrollTop = chatWin.scrollHeight;

      try {
        const payload = { message: msg };
        if (currentConversationId) payload.conversation_id = currentConversationId;
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        document.getElementById(typingId)?.remove();

        if (data.conversation_id && data.conversation_id !== 'new') {
          currentConversationId = data.conversation_id;
          const label = document.getElementById('chat-conv-label');
          if (label) label.textContent = 'Đang tiếp tục · ' + String(data.conversation_id).slice(0, 8);
          // Invalidate history cache so next open refreshes.
          const panel = document.getElementById('chat-history-panel');
          if (panel) panel.dataset.loaded = '0';
        }

        chatWin.innerHTML += renderAssistantBubble(
          data.content || '',
          data.content_html || null,
          data.citations || []
        );
        chatWin.scrollTop = chatWin.scrollHeight;

        // Agent trace (Dify style: thought → tool → observation)
        if ((data.steps && data.steps.length > 0) || (data.tool_calls && data.tool_calls.length > 0)) {
          chatWin.innerHTML += renderAgentTrace(data.steps, data.tool_calls);
          chatWin.scrollTop = chatWin.scrollHeight;
        }
      } catch (e) {
        document.getElementById(typingId)?.remove();
        chatWin.innerHTML += `<div class="text-red-400 text-xs ml-1">Lỗi: ${escapeHtml(e.message)}</div>`;
      }
    }

    // ========== MEMORY GRAPH (Brain Canvas) ==========
    const TYPE_COLORS = {
      rss: '#f59e0b', web: '#38bdf8', obsidian: '#a78bfa',
      manual: '#22d3ee', knowledge: '#34d399', project: '#fbbf24',
      writer_generated: '#c084fc', writer_duplicate_reference: '#fb7185',
    };
    const TYPE_LABELS = {
      rss: 'RSS', web: 'Web', obsidian: 'Obsidian', manual: 'Thủ công',
      knowledge: 'Kiến thức', project: 'Dự án', writer_generated: 'Viết AI',
      writer_duplicate_reference: 'Tham chiếu',
    };
    let graphData = null;
    let brainSim = null;

    function memoryTypeColor(type) { return TYPE_COLORS[type] || '#71717a'; }

    async function loadMemoryGraph() {
      const canvas = document.getElementById('brain-canvas');
      try {
        const res = await fetch('/api/memory/graph');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        graphData = data;
        // Stats panel
        renderMemoryStats(document.getElementById('memory-stats'), data);
        buildLegendAndFilters(data);
        if (!canvas) {
          showToast('Không tìm thấy canvas', 'error');
          return;
        }
        if (brainSim) brainSim.destroy();
        brainSim = new BrainSim(canvas, data);
      } catch (e) {
        if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        showToast('Không tải được đồ thị: ' + e.message, 'error');
      }
    }

    function buildLegendAndFilters(data) {
      const legend = document.getElementById('graph-legend');
      const filter = document.getElementById('graph-filter');
      const types = {};
      (data.nodes || []).forEach(n => { types[n.type] = (types[n.type] || 0) + 1; });
      legend.innerHTML = Object.keys(types).map(t =>
        `<span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full" style="background:${memoryTypeColor(t)}"></span>${TYPE_LABELS[t] || t} (${types[t]})</span>`
      ).join('') || '<span>Chưa có nút</span>';

      const typeKeys = Object.keys(types);
      filter.innerHTML = [`<button class="chip chip-active" data-t="" onclick="setGraphFilter(this,'')">Tất cả</button>`].concat(
        typeKeys.slice(0, 8).map(t =>
          `<button class="chip" data-t="${t}" onclick="setGraphFilter(this,'${t}')">${TYPE_LABELS[t] || t}</button>`
        )
      ).join('');
    }

    function setGraphFilter(btn, type) {
      document.querySelectorAll('#graph-filter .chip').forEach(c => c.classList.remove('chip-active'));
      btn.classList.add('chip-active');
      if (brainSim) brainSim.setFilter(type);
    }

    function renderMemoryGraph(container, data) {
      // Legacy placeholder — actual rendering is the BrainSim canvas.
      if (brainSim) brainSim.setData(data);
    }

    function renderMemoryStats(el, data) {
      const stats = (data && data.stats) || {};
      const byType = stats.by_type || {};
      el.innerHTML = `
        <div class="flex justify-between"><span class="text-zinc-400">Nguồn</span><span class="metric font-mono">${stats.sources || 0}</span></div>
        <div class="flex justify-between"><span class="text-zinc-400">Liên kết</span><span class="metric font-mono">${stats.edges || 0}</span></div>
        <div class="pt-2 border-t border-zinc-800 text-[10px] text-zinc-500">Các loại: ${Object.keys(byType).map(k => `${TYPE_LABELS[k] || k} · ${byType[k]}`).join(', ') || '—'}</div>
      `;
    }

    function showNodeDetail(id) {
      // Nút chi tiết hiển thị trong panel node-detail (bấm trên canvas).
      const node = (graphData.nodes || []).find(n => n.id === id);
      if (!node) return;
      const el = document.getElementById('node-detail');
      const type = node.type || 'manual';
      const children = (graphData.nodes || []).filter(n => n.parent_id === id);
      let html = `
        <div class="flex items-center gap-2 mb-2">
          <span class="w-2.5 h-2.5 rounded-full" style="background:${memoryTypeColor(type)}"></span>
          <div class="font-medium text-zinc-100 text-sm line-clamp-2">${escapeHtml(node.label || 'Không tên')}</div>
        </div>
        <div class="text-zinc-500 mb-1">${TYPE_LABELS[type] || type}${node.parent_id ? ' · <span class="text-amber-400">con của #' + escapeHtml((node.parent_title || '?').slice(0, 30)) + '</span>' : ''}</div>
        ${node.summary ? `<div class="text-zinc-300 mb-3 leading-relaxed">${escapeHtml(node.summary)}</div>` : ''}
      `;
      if (node.degree != null) html += `<div class="text-zinc-500 mb-2">Bậc: <span class="metric font-mono">${node.degree}</span> liên kết</div>`;
      if (node.tags && node.tags.length) html += `<div class="flex flex-wrap gap-1.5 mb-3">${node.tags.slice(0, 8).map(t => `<span class="px-2 py-0.5 rounded-full bg-zinc-900 border border-zinc-800 text-[10px]">${escapeHtml(t)}</span>`).join('')}</div>`;
      if (children.length) html += `<div class="text-[10px] text-zinc-500 mb-1">Con: ${children.slice(0, 8).map(c => escapeHtml(c.label.slice(0, 30))).join(', ') || '—'}</div>`;
      // 5 cạnh mạnh nhất
      const edges = (graphData.edges || []).map(e => ({ s: e.source, t: e.target, w: e.weight || 0, r: e.relation_type }))
        .filter(e => e.s === id || e.t === id).sort((a, b) => b.w - a.w).slice(0, 5);
      if (edges.length) {
        html += `<div class="text-[10px] text-zinc-500 mt-2 mb-1">Liên kết mạnh nhất:</div>`;
        edges.forEach(e => {
          const otherId = e.s === id ? e.t : e.s;
          const other = (graphData.nodes || []).find(n => n.id === otherId);
          html += `<div class="flex items-center gap-2 text-[11px] py-0.5">
            <span class="text-emerald-400 font-mono">${Math.round((e.w || 0) * 100)}%</span>
            <span class="text-zinc-300 truncate">${escapeHtml((other && other.label) || '?')}</span>
            <span class="text-zinc-600 ml-auto">${escapeHtml(e.r)}</span>
          </div>`;
        });
      }
      html += `<button onclick="resetGraphView()" class="mt-3 px-3 py-1 text-[11px] rounded-lg border border-zinc-700 hover:bg-zinc-800">Đóng</button>`;
      el.innerHTML = html;
    }

    function resetGraphView() {
      if (brainSim) { brainSim.selected = null; brainSim.resetView(); }
      document.getElementById('node-detail').innerHTML =
        '<i class="fa-solid fa-circle-info text-zinc-600 mr-1"></i> Bấm vào một nút trên đồ thị để xem chi tiết liên kết.';
    }

    // ========== Force-Directed Engine ==========
    class BrainSim {
      constructor(canvas, data) {
        this.cv = canvas;
        this.ctx = cv2d(canvas);
        this.nodes = [];
        this.edges = [];
        this.selected = null;
        this.hover = null;
        this.pan = { x: 0, y: 0 };
        this.zoom = 1;
        this.dragging = false;
        this.dragNode = null;
        this.width = canvas.width;
        this.height = canvas.height;
        this.setData(data);
        this.bindEvents();
        this.resizeBind = () => this.onResize();
        window.addEventListener('resize', this.resizeBind);
        this.raf = null;
        this.step();
        this._resizeTimer = setInterval(() => {
          if (!this.cv.isConnected || this.cv.clientWidth < 1) return;
          const dpr = window.devicePixelRatio || 1;
          const cw = this.cv.clientWidth;
          const ch = this.cv.clientHeight;
          const nextW = Math.max(300, Math.round(cw * dpr));
          const nextH = Math.max(300, Math.round(ch * dpr));
          if (Math.abs(this.cv.width - nextW) > 4 || Math.abs(this.cv.height - nextH) > 4) {
            this.onResize();
          }
        }, 300);
      }

      setFilter(type) {
        const filter = t => !type || t === type;
        this.nodes.forEach(n => { n.visible = filter(n.type); });
        this.edges.forEach(e => {
          e.visible = filter(e.sourceType) && filter(e.targetType);
        });
        this.pulseLayout(true);
      }

      setData(data) {
        this.data = data;
        const w = data.width || this.width;
        const h = data.height || this.height;
        const nodes = (data.nodes || []).map((n, i) => {
          const type = n.type || 'manual';
          const angle = (i / Math.max((data.nodes || []).length, 1)) * Math.PI * 2;
          const rad = Math.min(w, h) * 0.36;
          return { ...n, type, visible: true, x: w / 2 + Math.cos(angle) * rad, y: h / 2 + Math.sin(angle) * rad, vx: 0, vy: 0, degree: n.degree || 0 };
        });
        const byId = {};
        nodes.forEach(n => byId[n.id] = n);
        const edges = (data.edges || []).map(e => {
          const sId = e.source, tId = e.target;
          const s = byId[sId], t = byId[tId];
          if (!s || !t) return null;
          const weight = Number(e.weight) || 0;
          return { ...e, weight, w: weight, sx: s, tx: t, sourceType: s.type, targetType: t.type, visible: true };
        }).filter(Boolean);
        this.nodes = nodes;
        this.edges = edges;
        this.height = data.height || this.height;
        const meta = (data.meta || {});
        if (meta.view) this.pan = meta.view.pan || this.pan;
        this.pulseLayout(true);
      }

      pulseLayout(soft) {
        const nodes = this.nodes;
        const centerX = this.width / 2, centerY = this.height / 2;
        nodes.forEach((n, i) => {
          if (soft) {
            // attract visible to centre ring, keep momentum
            n.vx += (centerX - n.x) * 0.0002;
            n.vy += (centerY - n.y) * 0.0002;
          }
        });
      }

      bindEvents() {
        const c = this.cv;
        const self = this;
        c.addEventListener('mousedown', e => {
          const { x, y } = this.toWorld(e);
          this.dragging = true;
          this.dragNode = this.hitTest(x, y);
          this.dragStart = { mx: e.clientX, my: e.clientY, px: this.pan.x, py: this.pan.y };
          this.cv.classList.add('cursor-grabbing');
        });
        window.addEventListener('mousemove', e => {
          if (this.dragging) {
            if (this.dragNode) {
              const { x, y } = this.toWorld(e);
              this.dragNode.x = x; this.dragNode.y = y;
              this.dragNode.vx = 0; this.dragNode.vy = 0;
            } else if (this.dragStart) {
              this.pan.x = this.dragStart.px + (e.clientX - this.dragStart.mx);
              this.pan.y = this.dragStart.py + (e.clientY - this.dragStart.my);
            }
          } else {
            const { x, y } = this.toWorld(e);
            this.hover = this.hitTest(x, y, 16);
            this.cv.style.cursor = this.hover ? 'pointer' : 'grab';
          }
        });
        window.addEventListener('mouseup', e => {
          if (this.dragNode && !this.dragMoved) {
            this.selected = this.dragNode;
            showNodeDetail(this.dragNode.id);
          }
          this.dragging = false;
          this.dragNode = null;
          this.cv.classList.remove('cursor-grabbing');
        });
        c.addEventListener('wheel', e => {
          e.preventDefault();
          const factor = e.deltaY > 0 ? 0.9 : 1.1;
          this.zoom = Math.max(0.2, Math.min(4, this.zoom * factor));
        }, { passive: false });
        c.addEventListener('dblclick', () => { this.selected = null; resetGraphView(); });
      }

      toWorldEvt(e) {
        const r = this.cv.getBoundingClientRect();
        return { x: (e.clientX - r.left) / r.width * this.cv.width, y: (e.clientY - r.top) / r.height * this.cv.height };
      }
      toWorld(e) {
        const p = this.toWorldEvt(e);
        return { x: (p.x - this.pan.x) / this.zoom, y: (p.y - this.pan.y) / this.zoom };
      }

      hitTest(x, y, pad = 12) {
        let best = null, bd = (pad || 12) * (this.zoom < 0.7 ? 2 : 1);
        for (const n of this.nodes) {
          if (!n.visible) continue;
          const d = Math.hypot(n.x - x, n.y - y);
          if (d < bd + Math.max(this.nodeRadius(n), 4) && (!best || d < bd)) { best = n; bd = d; }
        }
        return best;
      }

      nodeRadius(n) { return 5 + Math.log2((n.degree || 0) + 1) * 4; }

      onResize() {
        const r = this.cv.getBoundingClientRect();
        const w = Math.max(300, Math.round((r.width || this.width || 640) * (window.devicePixelRatio || 1)));
        const h = Math.max(300, Math.round((r.height || this.height || 420) * (window.devicePixelRatio || 1)));
        this.cv.width = w;
        this.cv.height = h;
        this.width = w; this.height = h;
      }

      resetView() { this.pan = { x: 0, y: 0 }; this.zoom = 1; }

      destroy() {
        if (this._resizeTimer) clearInterval(this._resizeTimer);
        if (this.raf) cancelAnimationFrame(this.raf);
        this.raf = null;
        window.removeEventListener('resize', this.resizeBind);
      }

      step() {
        this.stepPhysics();
        this.draw();
        this.raf = requestAnimationFrame(() => this.step());
      }

      stepPhysics() {
        const nodes = this.nodes;
        const dt = 1;
        const repulse = 600, spring = 0.003, damping = 0.96;
        const w = this.width, h = this.height, cx = w / 2, cy = h / 2;
        const k = 1 + (nodes.length / 120);
        for (let i = 0; i < nodes.length; i++) {
          const a = nodes[i]; if (!a.visible) continue;
          for (let j = i + 1; j < nodes.length; j++) {
            const b = nodes[j]; if (!b.visible) continue;
            let dx = a.x - b.x, dy = a.y - b.y;
            let d2 = dx * dx + dy * dy || 1;
            let d = Math.sqrt(d2);
            let F = repulse / d2;
            a.vx += (dx / d) * F; a.vy += (dy / d) * F;
            b.vx -= (dx / d) * F; b.vy -= (dy / d) * F;
          }
          // spring to centre
          a.vx += (cx - a.x) * spring * 0.6 * k;
          a.vy += (cy - a.y) * spring * 0.6 * k;
        }
        for (const e of this.edges) {
          if (!e.visible || !e.sx.visible || !e.tx.visible) continue;
          const dx = e.tx.x - e.sx.x, dy = e.tx.y - e.sx.y;
          const dist = Math.hypot(dx, dy) || 1;
          const target = 115, force = (dist - target) * spring * (0.5 + e.w * 2);
          const fx = (dx / dist) * force, fy = (dy / dist) * force;
          e.sx.vx += fx; e.sx.vy += fy;
          e.tx.vx -= fx; e.tx.vy -= fy;
        }
        for (const n of nodes) {
          if (n.selected !== undefined) { /* no-op */ }
          n.vx = Math.max(-8, Math.min(8, n.vx));
          n.vy = Math.max(-8, Math.min(8, n.vy));
          n.x += n.vx * dt; n.y += n.vy * dt;
          n.vx *= damping; n.vy *= damping;
          // keep in-bounds loosely
          if (n.x < -200) n.x = -200; if (n.x > w + 200) n.x = w + 200;
          if (n.y < -200) n.y = -200; if (n.y > h + 200) n.y = h + 200;
        }
      }

      draw() {
        const ctx = this.ctx, cv = this.cv;
        
        // Beautiful dark background with subtle gradient
        const grad = ctx.createRadialGradient(cv.width/2, cv.height/2, 0, cv.width/2, cv.height/2, Math.max(cv.width, cv.height));
        grad.addColorStop(0, '#0f0e13');
        grad.addColorStop(1, '#0a0a0f');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, cv.width, cv.height);
        
        ctx.save();
        ctx.translate(this.pan.x, this.pan.y);
        ctx.scale(this.zoom, this.zoom);
        
        const selectedId = this.selected ? this.selected.id : null;
        const hoverId = this.hover ? this.hover.id : null;
        const time = Date.now() / 1000; // Animation time
        
        // ===== ANIMATED EDGES with flowing particles =====
        for (const e of this.edges) {
          if (!e.visible) continue;
          const w = 0.8 + (e.weight || 0) * 2.5;
          let alpha = 0.2 + (e.weight || 0) * 0.6;
          
          if (this.selected) {
            const connected = e.sx.id === selectedId || e.tx.id === selectedId;
            alpha = connected ? 0.95 : 0.08;
          }
          
          const related = e.w === 1 && e.relation_type === 'parent_child';
          
          // Calculate bezier curve
          const mx = (e.sx.x + e.tx.x) / 2, my = (e.sx.y + e.tx.y) / 2;
          const dx = e.tx.x - e.sx.x, dy = e.tx.y - e.sx.y;
          const len = Math.hypot(dx, dy) || 1;
          const bend = 0.25;
          const cx2 = mx + (-dy / len) * bend * (e.parentChild ? 26 : 14);
          
          // Base line with gradient
          ctx.beginPath();
          ctx.moveTo(e.sx.x, e.sx.y);
          ctx.bezierCurveTo(cx2, my, cx2, my, e.tx.x, e.tx.y);
          
          // Create gradient along line
          const lineGrad = ctx.createLinearGradient(e.sx.x, e.sx.y, e.tx.x, e.tx.y);
          if (related) {
            lineGrad.addColorStop(0, `rgba(251,191,36,${alpha * 0.6})`);
            lineGrad.addColorStop(0.5, `rgba(251,191,36,${alpha})`);
            lineGrad.addColorStop(1, `rgba(251,191,36,${alpha * 0.6})`);
          } else {
            lineGrad.addColorStop(0, `rgba(59,130,246,${alpha * 0.5})`);
            lineGrad.addColorStop(0.5, `rgba(139,92,246,${alpha * 0.8})`);
            lineGrad.addColorStop(1, `rgba(236,72,153,${alpha * 0.5})`);
          }
          
          ctx.strokeStyle = lineGrad;
          ctx.lineWidth = Math.max(0.6, w);
          ctx.lineCap = 'round';
          ctx.shadowBlur = 8;
          ctx.shadowColor = related ? 'rgba(251,191,36,0.4)' : 'rgba(139,92,246,0.3)';
          ctx.stroke();
          ctx.shadowBlur = 0;
          
          // Flowing particles on connected edges
          if (this.selected && (e.sx.id === selectedId || e.tx.id === selectedId)) {
            const numParticles = 3;
            for (let i = 0; i < numParticles; i++) {
              const offset = (i / numParticles + time * 0.3) % 1;
              const t = offset;
              
              // Bezier curve point calculation
              const x = Math.pow(1-t, 2) * e.sx.x + 
                       2 * (1-t) * t * cx2 + 
                       Math.pow(t, 2) * e.tx.x;
              const y = Math.pow(1-t, 2) * e.sx.y + 
                       2 * (1-t) * t * my + 
                       Math.pow(t, 2) * e.tx.y;
              
              // Glowing particle
              const particleGrad = ctx.createRadialGradient(x, y, 0, x, y, 4);
              particleGrad.addColorStop(0, 'rgba(255,255,255,0.9)');
              particleGrad.addColorStop(0.5, related ? 'rgba(251,191,36,0.6)' : 'rgba(139,92,246,0.6)');
              particleGrad.addColorStop(1, 'rgba(0,0,0,0)');
              
              ctx.fillStyle = particleGrad;
              ctx.beginPath();
              ctx.arc(x, y, 4, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        }
        
        // ===== BEAUTIFUL NODES with glow and rings =====
        for (const n of this.nodes) {
          if (!n.visible) continue;
          
          const r = this.nodeRadius(n);
          const col = memoryTypeColor(n.type);
          const isSel = n.id === selectedId, isHov = n.id === hoverId;
          
          const nodeAlpha = this.selected ? (isSel || this.isConnected(n, selectedId) ? 1 : 0.25) : (isHov ? 1 : 0.85);
          ctx.globalAlpha = nodeAlpha;
          
          // Outer glow
          if (isSel || isHov) {
            const glowSize = r + 12 + Math.sin(time * 3) * 2;
            const glowGrad = ctx.createRadialGradient(n.x, n.y, r, n.x, n.y, glowSize);
            glowGrad.addColorStop(0, 'rgba(0,0,0,0)');
            glowGrad.addColorStop(0.7, col.replace(')', ',0.3)').replace('rgb', 'rgba'));
            glowGrad.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = glowGrad;
            ctx.beginPath();
            ctx.arc(n.x, n.y, glowSize, 0, Math.PI * 2);
            ctx.fill();
          }
          
          // Pulsing ring for selected/hover
          if (isSel) {
            const pulseR = r + 6 + Math.sin(time * 4) * 3;
            ctx.beginPath();
            ctx.arc(n.x, n.y, pulseR, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255,255,255,0.6)';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            const pulseR2 = r + 10 + Math.sin(time * 4 + 0.5) * 3;
            ctx.beginPath();
            ctx.arc(n.x, n.y, pulseR2, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.lineWidth = 1;
            ctx.stroke();
          }
          
          // Main node with gradient
          const nodeGrad = ctx.createRadialGradient(
            n.x - r * 0.3, n.y - r * 0.3, 0,
            n.x, n.y, r
          );
          
          // Parse color and create lighter version for gradient
          const brightCol = col.replace(/rgba?\((\d+),\s*(\d+),\s*(\d+)/, (m, r, g, b) => {
            return `rgb(${Math.min(255, parseInt(r) + 40)},${Math.min(255, parseInt(g) + 40)},${Math.min(255, parseInt(b) + 40)})`;
          });
          
          nodeGrad.addColorStop(0, brightCol);
          nodeGrad.addColorStop(0.7, col);
          nodeGrad.addColorStop(1, col.replace(')', ',0.8)').replace('rgb', 'rgba'));
          
          ctx.fillStyle = nodeGrad;
          ctx.shadowBlur = isSel || isHov ? 16 : 8;
          ctx.shadowColor = col;
          ctx.beginPath();
          ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
          
          // Outer ring
          ctx.beginPath();
          ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
          ctx.lineWidth = isSel ? 2.5 : (isHov ? 2 : 1.2);
          const ringAlpha = isSel ? 0.95 : (isHov ? 0.7 : 0.4);
          ctx.strokeStyle = `rgba(255,255,255,${ringAlpha})`;
          ctx.stroke();
          
          // Inner highlight
          ctx.beginPath();
          ctx.arc(n.x - r * 0.35, n.y - r * 0.35, r * 0.3, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255,255,255,0.3)';
          ctx.fill();
          
          ctx.globalAlpha = 1;
        }
        
        // ===== LABELS with shadow =====
        const minZoom = 0.55;
        if (this.zoom >= minZoom) {
          ctx.font = `600 ${11}px system-ui, -apple-system, sans-serif`;
          ctx.textAlign = 'center';
          
          for (const n of this.nodes) {
            if (!n.visible) continue;
            if (this.selected && n.id !== selectedId && !this.isConnected(n, selectedId)) continue;
            
            const label = (n.label || '').slice(0, 30);
            const y = n.y + this.nodeRadius(n) + 14;
            
            // Text shadow for readability
            ctx.shadowBlur = 4;
            ctx.shadowColor = 'rgba(0,0,0,0.8)';
            ctx.fillStyle = 'rgba(228,228,231,0.95)';
            ctx.fillText(label, n.x, y);
            ctx.shadowBlur = 0;
          }
        }
        
        ctx.restore();
        
        // ===== OVERLAY INFO with gradient background =====
        const overlayGrad = ctx.createLinearGradient(0, 0, 200, 0);
        overlayGrad.addColorStop(0, 'rgba(0,0,0,0.6)');
        overlayGrad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = overlayGrad;
        ctx.fillRect(0, 0, 200, 32);
        
        ctx.fillStyle = 'rgba(163,163,163,0.7)';
        ctx.font = '11px system-ui, sans-serif';
        ctx.textAlign = 'left';
        const visibleNodes = this.nodes.filter(n => n.visible).length;
        const visibleEdges = this.edges.filter(e => e.visible).length;
        ctx.fillText(`${visibleNodes} nút · ${visibleEdges} liên kết`, 14, 20);
      }

      isConnected(n, id) {
        return this.edges.some(e => (e.sx.id === n.id && e.tx.id === id) || (e.tx.id === n.id && e.sx.id === id));
      }
    }

    function cv2d(canvas) {
      const dpr = (window.devicePixelRatio || 1);
      const cw = canvas.clientWidth || canvas.offsetWidth || 640;
      const ch = canvas.clientHeight || canvas.offsetHeight || 420;
      canvas.width = Math.max(300, Math.round(cw * dpr));
      canvas.height = Math.max(300, Math.round(ch * dpr));
      return canvas.getContext('2d');
    }

    // ========== INGEST ==========
    async function handleFileUpload(input) {
      const file = input.files[0];
      if (!file) return;
      const resultBox = document.getElementById('ingest-result');
      resultBox.classList.remove('hidden');
      resultBox.innerHTML = ``;
      startIngestPipeline();
      try {
        setPipeStep(0, 'active');
        logIngest(`Đang tải lên ${file.name}...`, 'info');
        const form = new FormData();
        form.append('file', file, file.name);
        const res = await fetch('/api/ingest/file', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(apiErrorMessage(data, 'Lỗi nạp file'));
        setPipeStepAll('done');
        updateNeuralProgress(100);
        setNeuralState('done', 'Hoàn tất', `Đã nạp ${data.added || 0} chunk mới`);
        resultBox.innerHTML = renderIngestResult(data, true);
        showToast('Nạp file thành công');
      } catch (e) {
        setPipeStepAll('fail');
        setNeuralState('error', 'Lỗi', e.message);
        resultBox.innerHTML = renderIngestResult({ error: e.message }, false);
      }
      stopIngestPipeline();
      input.value = '';
    }

    async function ingestText() {
      const text = document.getElementById('ingest-text').value.trim();
      const title = document.getElementById('ingest-title')?.value?.trim() || '';
      if (!text) return showToast('Vui lòng nhập nội dung', 'error');
      await ingestTextDirect(text, title || undefined);
    }

    async function ingestTextDirect(content, title) {
      const resultBox = document.getElementById('ingest-result');
      resultBox.classList.remove('hidden');
      resultBox.innerHTML = ``;
      startIngestPipeline();
      try {
        await wait(260);
        setPipeStep(0, 'active');
        logIngest('Nhận dữ liệu đầu vào...', 'info');

        const body = { content, source_type: 'manual' };
        if (title) body.title = title;
        const res = await fetch('/api/ingest/text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(apiErrorMessage(data, 'Lỗi nạp kiến thức'));

        setPipeStepAll('done');
        updateNeuralProgress(100);
        setNeuralState('done', 'Hoàn tất', `Đã nạp ${data.added || 0} chunk mới`);

        resultBox.innerHTML = renderIngestResult(data, true);
        showToast('Nạp kiến thức thành công');
      } catch (e) {
        setPipeStepAll('fail');
        setNeuralState('error', 'Lỗi', e.message);
        resultBox.innerHTML = renderIngestResult({ error: e.message }, false);
      }
      stopIngestPipeline();
    }

    function renderIngestResult(data, ok) {
      if (!ok) {
        return `<div class="ingest-result-card error">
          <div class="flex items-center gap-2 text-red-400 font-medium"><i class="fa-solid fa-triangle-exclamation"></i> Lỗi nạp kiến thức</div>
          <div class="text-zinc-400 text-sm mt-1">${escapeHtml(data.error || 'Không rõ nguyên nhân')}</div>
        </div>`;
      }
      return `<div class="ingest-result-card">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2 text-emerald-400 font-medium"><i class="fa-solid fa-check-circle"></i> Đã nạp thành công</div>
          <div class="text-[10px] text-zinc-500 font-mono">${escapeHtml(data.source || '—')}</div>
        </div>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <div class="stat-chip"><i class="fa-solid fa-plus text-emerald-400"></i><b>${data.added || 0}</b> thêm mới</div>
          <div class="stat-chip"><i class="fa-solid fa-rotate text-cyan-400"></i><b>${data.updated || 0}</b> cập nhật</div>
          <div class="stat-chip"><i class="fa-solid fa-copy text-zinc-500"></i><b>${data.skipped || 0}</b> trùng bỏ qua</div>
          <div class="stat-chip"><i class="fa-solid fa-triangle-exclamation text-amber-400"></i><b>${(data.errors || []).length}</b> lỗi</div>
        </div>
        ${(data.errors && data.errors.length) ? `<div class="mt-3 text-xs text-amber-400">${data.errors.slice(0,5).map(escapeHtml).join('<br>')}</div>` : ''}
      </div>`;
    }

    // ========== WRITER ==========
    let writerOptionsLoaded = false;
    const writerContentCache = {};  // content_id -> full content string

    async function loadWriterOptions() {
      if (writerOptionsLoaded) return;
      try {
        const res = await fetch('/api/ui/writer-options');
        const opt = await res.json();
        const fill = (id, items) => {
          const sel = document.getElementById(id);
          if (!sel) return;
          sel.innerHTML = items.map(it =>
            `<option value="${escapeHtml(it.meta?.value || it.key)}">${escapeHtml(it.label)}</option>`
          ).join('');
        };
        fill('writer-type', opt.content_type);
        fill('writer-tone', opt.tone);
        fill('writer-audience', opt.audience);
        fill('writer-language', opt.language);
        const maxN = 3;
        const variants = document.getElementById('writer-variants');
        if (variants) {
          variants.innerHTML = Array.from({ length: maxN }, (_, i) =>
            `<option value="${i + 1}">${i + 1} bài</option>`).join('');
        }
        writerOptionsLoaded = true;
      } catch (e) {
        showToast('Không tải được tùy chọn writer', 'error');
      }
    }

    async function loadWriterHistory() {
      const list = document.getElementById('writer-history-list');
      if (!list) return;
      list.innerHTML = '<div class="text-xs text-zinc-500 px-1">Đang tải lịch sử...</div>';
      try {
        const res = await fetch('/api/writer/content?limit=30');
        const data = await res.json();
        const items = Array.isArray(data) ? data : (data.items || []);
        if (!items.length) {
          list.innerHTML = '<div class="text-xs text-zinc-500 px-1">Chưa có bài viết nào.</div>';
          return;
        }
        list.innerHTML = items.map(it => {
          const title = escapeHtml(it.topic || 'Không tiêu đề');
          const date = it.created_at ? new Date(it.created_at).toLocaleString('vi-VN', { day:'2-digit', month:'2-digit', year:'numeric', hour:'2-digit', minute:'2-digit' }) : '';
          return `<button onclick="openWriterContent('${escapeHtml(it.content_id)}')" class="w-full text-left p-3 rounded-2xl hover:bg-zinc-900/70 border border-zinc-800/50 hover:border-zinc-700 transition-all">
            <div class="text-sm text-zinc-200 truncate">${title}</div>
            <div class="text-[10px] text-zinc-500 mt-1">${date}</div>
          </button>`;
        }).join('');
      } catch (e) {
        list.innerHTML = `<div class="text-xs text-red-400 px-1">Lỗi: ${escapeHtml(e.message)}</div>`;
      }
    }

    async function openWriterContent(contentId) {
      const resultBox = document.getElementById('writer-result');
      const empty = document.getElementById('writer-empty');
      resultBox.classList.remove('hidden');
      if (empty) empty.classList.add('hidden');
      resultBox.innerHTML = `<div class="glass rounded-3xl p-6 border border-zinc-800 flex items-center gap-3 text-sm"><i class="fa-solid fa-spinner fa-spin"></i> Đang mở bài viết...</div>`;
      try {
        const res = await fetch(`/api/writer/content/${encodeURIComponent(contentId)}`);
        if (!res.ok) throw new Error(apiErrorMessage(await res.json().catch(() => ({})), 'Lỗi tải bài viết'));
        const item = await res.json();
        writerContentCache[item.content_id] = item.content;
        resultBox.innerHTML = renderVariantCard({ ...item, meta: item.meta || {} }, 0);
      } catch (e) {
        resultBox.innerHTML = `<div class="glass rounded-3xl p-6 border border-red-900/40 text-red-400 text-sm">Lỗi: ${escapeHtml(e.message)}</div>`;
      }
    }

    function dupBadgeColor(pct) {
      if (pct >= 80) return 'text-red-400 border-red-900/50 bg-red-950/30';
      if (pct >= 65) return 'text-amber-400 border-amber-900/50 bg-amber-950/30';
      return 'text-emerald-400 border-emerald-900/50 bg-emerald-950/30';
    }

    function renderDuplicatePanel(check) {
      if (!check) return '';
      const status = check.status || 'unknown';
      if (status === 'skipped') {
        return `<div class="mt-4 pt-4 border-t border-zinc-800 text-xs text-zinc-500">Kiểm tra trùng web: đã tắt.</div>`;
      }
      const matches = check.matches || [];
      const highest = check.highest_similarity_percent || 0;
      const highestSemantic = check.highest_semantic_percent;
      const pageCount = (check.pages || []).length;
      let statusNote = '';
      if (status === 'unavailable') statusNote = `<span class="text-zinc-500">(không truy cập được web)</span>`;
      else if (status === 'partial') statusNote = `<span class="text-amber-400">(một phần — có lỗi khi tìm kiếm)</span>`;

      let html = `<div class="mt-4 pt-4 border-t border-zinc-800">
        <div class="flex items-center gap-2 text-xs mb-2">
          <i class="fa-solid fa-globe text-zinc-500"></i>
          <span class="text-zinc-400">Trùng lặp trên web</span>
          ${statusNote}
          <span class="ml-auto font-mono ${highest >= 65 ? 'text-amber-400' : 'text-emerald-400'}">từ vựng ${highest}%</span>
        </div>
        <div class="text-[10px] text-zinc-600 mb-2">${pageCount} trang đã kiểm tra${highestSemantic == null ? '' : ` · ngữ nghĩa ${highestSemantic}%`}</div>`;
      if (matches.length === 0) {
        html += `<div class="text-emerald-400 text-xs">✓ Không tìm thấy trang trùng đáng kể.</div>`;
      } else {
        matches.forEach(m => {
          const pct = m.similarity_percent || 0;
          const badge = dupBadgeColor(pct);
          html += `<div class="flex items-center gap-3 py-2 border-b border-zinc-800/60 last:border-0">
            <div class="flex-1 min-w-0">
              <a href="${escapeHtml(m.url)}" target="_blank" rel="noopener" class="text-xs text-blue-400 hover:underline block truncate">${escapeHtml(m.title || m.url)}</a>
              <div class="text-[10px] text-zinc-600 truncate">${escapeHtml(m.url)}</div>
            </div>
            <div class="text-right">
              <div class="text-[10px] px-2 py-1 rounded-lg border font-mono ${badge}">${pct}%</div>
              ${m.semantic_percent == null ? '' : `<div class="text-[10px] text-zinc-500 mt-1">ngữ nghĩa ${m.semantic_percent}%</div>`}
            </div>
          </div>`;
        });
      }
      html += `</div>`;
      return html;
    }

    function renderVariantCard(item, idx) {
      writerContentCache[item.content_id] = item.content;
      const meta = item.meta || {};
      const check = meta.web_duplicate_check;
      const ingestStatus = meta.ingest_status;
      const full = escapeHtml(item.content);
      const referenceCount = meta.ingest_reference_page_count || (check && check.pages ? check.pages.filter(page => page.fetched).length : 0);
      let badge = '';
      if (ingestStatus === 'ingested') {
        badge += `<span class="text-[10px] px-2 py-1 rounded-lg border border-emerald-900/50 text-emerald-400 bg-emerald-950/30"><i class="fa-solid fa-brain mr-1"></i>Đã nạp brain</span>`;
      } else if (ingestStatus) {
        badge += `<span class="text-[10px] px-2 py-1 rounded-lg border border-zinc-700 text-zinc-500">${escapeHtml(ingestStatus)}</span>`;
      }
      if (referenceCount) {
        badge += `<span class="text-[10px] px-2 py-1 rounded-lg border border-sky-900/50 text-sky-400 bg-sky-950/30"><i class="fa-solid fa-magnifying-glass-chart mr-1"></i>Đã học từ ${referenceCount} trang web</span>`;
      }
      return `<div class="glass rounded-3xl p-6 border border-zinc-800">
        <div class="flex items-center gap-2 mb-3">
          <span class="text-xs text-zinc-400 font-medium">Bài ${idx + 1}</span>
          <span class="text-[10px] text-zinc-600 font-mono">${escapeHtml(item.content_id)}</span>
          <div class="ml-auto flex items-center gap-2">${badge}</div>
        </div>
        <h3 class="text-base font-semibold text-zinc-100 mb-3">${escapeHtml(item.topic || 'Bài viết')}</h3>
        <div class="prose prose-invert max-w-none text-sm whitespace-pre-wrap">${full}</div>
        <div class="mt-4 pt-4 border-t border-zinc-800 flex gap-3">
          <button onclick="copyToClipboard('${item.content_id}')" class="text-xs px-4 py-2 rounded-xl border border-zinc-700 hover:bg-zinc-900">Copy</button>
          <button onclick="refineContent('${item.content_id}')" class="text-xs px-4 py-2 rounded-xl border border-zinc-700 hover:bg-zinc-900">Chỉnh sửa</button>
          <button onclick="recheckExternal('${item.content_id}', this)" class="text-xs px-4 py-2 rounded-xl border border-blue-800/60 text-blue-400 hover:bg-blue-950/40"><i class="fa-solid fa-magnifying-glass mr-1"></i>Kiểm tra trùng web</button>
        </div>
        <div class="dup-panel">${renderDuplicatePanel(check)}</div>
      </div>`;
    }

    async function generateContent() {
      const type = document.getElementById('writer-type').value;
      const topic = document.getElementById('writer-topic').value.trim();
      const keywords = document.getElementById('writer-keywords').value.split(',').map(s => s.trim()).filter(Boolean);
      const tone = document.getElementById('writer-tone').value;
      const audience = document.getElementById('writer-audience').value;
      const raw_wc = parseInt(document.getElementById('writer-wordcount').value, 10);
      const word_count = Number.isFinite(raw_wc) && raw_wc >= 100 && raw_wc <= 3000 ? raw_wc : 600;
      const output_format = 'plain';
      const language = document.getElementById('writer-language').value;
      const article_count = parseInt(document.getElementById('writer-variants').value, 10) || 1;
      const check_web_duplicates = false;
      const ingest_to_brain = false;

      if (!topic) return showToast('Vui lòng nhập gợi ý tiêu đề hoặc chủ đề', 'error');

      const btn = document.getElementById('writer-generate-btn');
      btn.disabled = true;
      const warnBox = document.getElementById('writer-warnings');
      warnBox.classList.add('hidden');
      warnBox.innerHTML = '';
      const resultBox = document.getElementById('writer-result');
      const empty = document.getElementById('writer-empty');
      if (empty) empty.classList.add('hidden');
      resultBox.classList.remove('hidden');
      resultBox.innerHTML = `<div class="glass rounded-3xl p-6 border border-zinc-800 flex items-center gap-3 text-sm"><i class="fa-solid fa-spinner fa-spin"></i> Đang tạo ${article_count} bài viết...</div>`;

      try {
        const res = await fetch('/api/writer/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content_type: type, topic, keywords, tone, audience,
            word_count, output_format, language, article_count,
            check_web_duplicates, ingest_to_brain, extra: {}
          })
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(apiErrorMessage(err, 'Lỗi tạo nội dung'));
        }
        const data = await res.json();
        const items = data.items || [];
        if (items.length === 0) {
          resultBox.innerHTML = `<div class="glass rounded-3xl p-6 border border-zinc-800 text-amber-400 text-sm">Không tạo được bài viết nào.</div>`;
        } else {
          resultBox.innerHTML = items.map((it, i) => renderVariantCard(it, i)).join('');
        }
        if (data.warnings && data.warnings.length) {
          warnBox.classList.remove('hidden');
          warnBox.innerHTML = `<div class="glass rounded-2xl p-4 border border-amber-900/40 text-xs text-amber-400">
            <div class="font-medium mb-1"><i class="fa-solid fa-triangle-exclamation mr-1"></i>Cảnh báo</div>
            ${data.warnings.map(w => `<div>• ${escapeHtml(w)}</div>`).join('')}
          </div>`;
        }
        const anyIngest = items.some(it => (it.meta || {}).ingest_status === 'ingested');
        showToast(anyIngest
          ? `Đã tạo ${data.completed_count} bài — braind đang học cách viết bài này.`
          : `Đã tạo ${data.completed_count} bài.`, 'success');
        loadWriterHistory();
      } catch (e) {
        resultBox.innerHTML = `<div class="glass rounded-3xl p-6 border border-red-900/40 text-red-400 text-sm">Lỗi: ${escapeHtml(e.message)}</div>`;
      } finally {
        btn.disabled = false;
      }
    }

    async function recheckExternal(id, btn) {
      const card = btn.closest('.glass');
      const panel = card ? card.querySelector('.dup-panel') : null;
      if (panel) panel.innerHTML = `<div class="mt-4 pt-4 border-t border-zinc-800 text-xs text-zinc-400"><i class="fa-solid fa-spinner fa-spin mr-2"></i>Đang kiểm tra web...</div>`;
      try {
        const res = await fetch('/api/writer/check_external', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content_id: id })
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(apiErrorMessage(err, 'Lỗi kiểm tra'));
        }
        const data = await res.json();
        if (panel) panel.innerHTML = renderDuplicatePanel(data.web_duplicate_check);
        if (data.references_learned > 0) {
          showToast(`Đã học từ ${data.references_learned} trang web và nạp vào braind.`, 'success');
        }
      } catch (e) {
        if (panel) panel.innerHTML = `<div class="mt-4 pt-4 border-t border-zinc-800 text-xs text-red-400">Lỗi: ${escapeHtml(e.message)}</div>`;
      }
    }

    async function checkSimilarity() {
      const topic = document.getElementById('writer-topic').value.trim();
      if (!topic) return showToast('Nhập chủ đề trước khi kiểm tra trùng', 'error');
      const panel = document.getElementById('similarity-panel');
      panel.classList.remove('hidden');
      panel.innerHTML = `<div class="glass p-5 rounded-2xl text-sm">Đang kiểm tra nội bộ...</div>`;
      try {
        const res = await fetch('/api/writer/check_similarity', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic, threshold: 0.72 })
        });
        const data = await res.json();
        let html = `<div class="glass p-5 rounded-2xl"><div class="text-xs text-zinc-400 mb-2">Trùng nội bộ: ${data.count} bài tương tự (≥${data.threshold})</div>`;
        if (data.matches.length === 0) html += `<div class="text-emerald-400 text-sm">✓ Không có nội dung nội bộ trùng đáng kể.</div>`;
        else data.matches.forEach(m => {
          html += `<div class="flex justify-between text-xs py-2 border-b border-zinc-800 last:border-0"><div>${escapeHtml(m.topic)}</div><div class="font-mono text-amber-400">${(m.similarity*100).toFixed(0)}%</div></div>`;
        });
        html += `</div>`;
        panel.innerHTML = html;
      } catch (e) {
        panel.innerHTML = `<div class="glass p-5 rounded-2xl text-red-400">Lỗi: ${escapeHtml(e.message)}</div>`;
      }
    }

    function copyToClipboard(id) {
      const content = writerContentCache[id] || '';
      if (!content) return showToast('Không có nội dung để copy', 'error');
      navigator.clipboard.writeText(content).then(
        () => showToast('Đã copy nội dung vào clipboard', 'success'),
        () => showToast('Trình duyệt chặn clipboard', 'error')
      );
    }

    function refineContent(id) {
      const instruction = prompt('Hướng dẫn chỉnh sửa:');
      if (!instruction) return;
      fetch('/api/writer/refine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content_id: id, instruction })
      }).then(r => r.json()).then(d => {
        showToast('Đã chỉnh sửa thành công');
        writerContentCache[id] = d.content;
      }).catch(e => showToast('Lỗi chỉnh sửa: ' + e.message, 'error'));
    }

    // ========== DEEP LEARNING ==========
    const learnStageIndex = { queries: 0, search: 1, history: 2, fetch: 3, analyze: 4, links: 5, ingest: 6 };
    let learnRunning = false;
    let learnPages = 0;

    function setLearnState(mode, text) {
      const shell = document.getElementById('learn-shell');
      const label = document.getElementById('learn-state-label');
      const badge = document.getElementById('learn-status-badge');
      if (!shell || !label || !badge) return;
      shell.classList.toggle('ingesting', mode === 'running');
      label.classList.remove('running', 'error', 'idle');
      if (mode === 'running') {
        label.classList.add('running'); label.textContent = '◉ LEARNING';
        badge.textContent = 'Đang tự học'; badge.className = 'text-xs font-mono uppercase tracking-wider px-4 py-2 rounded-full border text-emerald-300 border-emerald-500/30 bg-emerald-500/10';
      } else if (mode === 'error') {
        label.classList.add('error'); label.textContent = '⚠ ERROR';
        badge.textContent = 'Có lỗi'; badge.className = 'text-xs font-mono uppercase tracking-wider px-4 py-2 rounded-full border text-red-300 border-red-500/30 bg-red-500/10';
      } else if (mode === 'done') {
        label.textContent = '✓ DONE';
        badge.textContent = 'Hoàn tất'; badge.className = 'text-xs font-mono uppercase tracking-wider px-4 py-2 rounded-full border text-emerald-300 border-emerald-500/30 bg-emerald-500/10';
      } else {
        label.classList.add('idle'); label.textContent = '⏻ IDLE';
        badge.textContent = 'Sẵn sàng'; badge.className = 'text-xs font-mono uppercase tracking-wider px-4 py-2 rounded-full border text-zinc-500 border-zinc-800';
      }
    }

    function setLearnStage(stage, state = 'active') {
      const i = learnStageIndex[stage];
      if (i === undefined) return;
      document.querySelectorAll('#learn-pipeline .pipe-step').forEach((el, index) => {
        el.classList.remove('active', 'done', 'fail', 'idle');
        if (index < i) el.classList.add('done');
        else if (index === i) el.classList.add(state);
      });
    }

    function resetLearnDashboard() {
      learnPages = 0;
      ['m-pages', 'm-ingested', 'm-duplicates', 'm-errors'].forEach(id => { const el = document.getElementById(id); if (el) el.textContent = '0'; });
      const pages = document.getElementById('learn-pages');
      if (pages) pages.replaceChildren();
      const count = document.getElementById('learn-page-count');
      if (count) count.textContent = '0';
      clearLearnLog();
      document.querySelectorAll('#learn-pipeline .pipe-step').forEach(el => el.classList.remove('active', 'done', 'fail'));
    }

    function clearLearnLog() {
      const body = document.getElementById('learn-log-body');
      if (body) body.replaceChildren();
    }

    function logLearn(text, level = 'info') {
      const body = document.getElementById('learn-log-body');
      if (!body) return;
      const row = document.createElement('div');
      row.className = level === 'error' ? 'err' : level;
      const stamp = new Date().toLocaleTimeString('vi-VN', { hour12: false });
      row.textContent = `[${stamp}] ${text || ''}`;
      body.appendChild(row);
      while (body.children.length > 220) body.removeChild(body.firstChild);
      body.scrollTop = body.scrollHeight;
    }

    function setLearnMetrics(summary) {
      const metricMap = { pages: 'm-pages', ingested: 'm-ingested', duplicates: 'm-duplicates', errors: 'm-errors' };
      Object.entries(metricMap).forEach(([key, id]) => {
        const el = document.getElementById(id);
        if (el && Number.isFinite(Number(summary?.[key]))) el.textContent = String(summary[key]);
      });
    }

    function appendLearnPage(page) {
      const body = document.getElementById('learn-pages');
      if (!body) return;
      learnPages += 1;
      const count = document.getElementById('learn-page-count');
      if (count) count.textContent = String(learnPages);
      const row = document.createElement('div');
      row.className = 'learn-page';
      const dot = document.createElement('span'); dot.className = `ldot ${page.status || 'queued'}`;
      const url = document.createElement('a');
      url.className = 'truncate text-sky-300 hover:text-sky-200'; url.target = '_blank'; url.rel = 'noreferrer';
      url.href = page.url || '#'; url.textContent = compactUrl(page.url || '');
      const title = document.createElement('span'); title.className = 'truncate text-zinc-300'; title.title = page.title || ''; title.textContent = page.title || page.error || 'Không có tiêu đề';
      const depth = document.createElement('span'); depth.className = 'text-zinc-500 text-center'; depth.textContent = `L${page.depth ?? 0}`;
      const status = document.createElement('span'); status.className = `learn-tag ${page.status || 'queued'}`; status.textContent = page.status || 'queued';
      row.append(dot, url, title, depth, status);
      body.prepend(row);
    }

    function compactUrl(value) {
      try {
        const url = new URL(value);
        return `${url.hostname}${url.pathname === '/' ? '' : url.pathname}`.slice(0, 58);
      } catch (_) { return String(value).slice(0, 58); }
    }

    function consumeLearnEvent(event, payload) {
      if (event === 'stage') {
        setLearnStage(payload.stage);
        if (payload.label) document.getElementById('learn-excerpt').textContent = `Đang thực hiện: ${payload.label}${payload.url ? ` — ${compactUrl(payload.url)}` : ''}`;
      } else if (event === 'log') {
        logLearn(payload.text || '', payload.level === 'err' ? 'error' : (payload.level || 'info'));
      } else if (event === 'page') {
        appendLearnPage(payload);
        setLearnMetrics({
          pages: learnPages,
          ingested: Number(document.getElementById('m-ingested').textContent) + Number(payload.ingested || 0),
          duplicates: Number(document.getElementById('m-duplicates').textContent) + Number(payload.status === 'duplicate'),
          errors: Number(document.getElementById('m-errors').textContent) + Number(payload.status === 'error'),
        });
      } else if (event === 'summary') {
        setLearnMetrics(payload);
        document.getElementById('learn-excerpt').textContent = `Đã hoàn tất: ${payload.pages || 0} trang, nạp ${payload.ingested || 0} dữ liệu, ${payload.skipped || 0} URL đã có lịch sử.`;
      } else if (event === 'error') {
        setLearnState('error');
        logLearn(payload.error || 'Lỗi không xác định.', 'error');
      } else if (event === 'done') {
        setLearnState('done');
        setLearnStage('ingest', 'done');
        logLearn('Chu trình tự học đã hoàn tất.', 'ok');
        loadLearnHistory();
      }
    }

    async function readLearnSSE(response, handler = consumeLearnEvent) {
      if (!response.body) throw new Error('Trình duyệt không hỗ trợ stream phản hồi.');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() || '';
        for (const frame of frames) {
          let event = 'message';
          const dataLines = [];
          for (const line of frame.split(/\r?\n/)) {
            if (line.startsWith('event:')) event = line.slice(6).trim();
            if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
          }
          if (!dataLines.length) continue;
          try { handler(event, JSON.parse(dataLines.join('\n'))); }
          catch (error) { console.warn('Invalid learning SSE frame', error); }
        }
        if (done) break;
      }
      if (buffer.trim()) {
        const data = buffer.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
        if (data) { try { handler('message', JSON.parse(data)); } catch (_) {} }
      }
    }

    // ========== SITE CRAWL (học cả một website) ==========
    let crawlController = null;

    function crawlLog(text, level = 'info') {
      const box = document.getElementById('crawl-log');
      if (!box) return;
      const colors = { ok: 'text-emerald-400', warn: 'text-amber-400', error: 'text-red-400', info: 'text-zinc-400' };
      const line = document.createElement('div');
      line.className = colors[level] || colors.info;
      line.textContent = text;
      box.appendChild(line);
      // Keep the DOM bounded: a 300-page crawl emits far more lines than a
      // reader will ever scroll back through.
      while (box.childElementCount > 200) box.removeChild(box.firstChild);
      box.scrollTop = box.scrollHeight;
    }

    function setCrawlRunning(running) {
      const start = document.getElementById('crawl-btn');
      const stop = document.getElementById('crawl-stop');
      const panel = document.getElementById('crawl-progress');
      if (start) { start.disabled = running; start.classList.toggle('loading', running); }
      if (stop) stop.classList.toggle('hidden', !running);
      if (panel && running) panel.classList.remove('hidden');
    }

    function consumeCrawlEvent(event, payload) {
      const stage = document.getElementById('crawl-stage');
      const counts = document.getElementById('crawl-counts');
      if (event === 'stage' && stage) {
        stage.textContent = payload.label || payload.stage || '';
      } else if (event === 'log') {
        crawlLog(payload.text || '', payload.level);
      } else if (event === 'page') {
        const mark = { success: '✓', duplicate: '≡', irrelevant: '·', short: '·', chinese: '·', error: '✕' }[payload.status] || '·';
        const level = payload.status === 'success' ? 'ok' : (payload.status === 'error' ? 'error' : 'warn');
        crawlLog(`${mark} d${payload.depth} ${(payload.title || payload.url || '').slice(0, 80)}`, level);
      } else if (event === 'summary' || event === 'done') {
        const s = payload.summary || payload;
        if (counts) counts.textContent = `${s.pages || 0} trang · ${s.ingested || 0} nạp · ${s.duplicates || 0} trùng · ${s.errors || 0} lỗi`;
        if (event === 'done') {
          if (stage) stage.textContent = s.timed_out ? 'Hết thời gian — đã lưu phần đã học' : 'Hoàn tất';
          crawlLog(`Xong: ${s.pages || 0} trang, sâu nhất ${s.max_depth_reached || 0} tầng, nạp ${s.ingested || 0} mục.`, 'ok');
          showToast(`Đã học ${s.ingested || 0} mục từ ${s.site || 'website'}`);
          loadLearnHistory();
        }
      } else if (event === 'error') {
        if (stage) stage.textContent = 'Lỗi';
        crawlLog(payload.message || 'Duyệt website thất bại.', 'error');
      }
    }

    async function startSiteCrawl() {
      const url = document.getElementById('crawl-url').value.trim();
      if (!url) return showToast('Vui lòng nhập URL website', 'error');
      if (!/^https?:\/\//i.test(url)) return showToast('URL phải bắt đầu bằng http:// hoặc https://', 'error');

      document.getElementById('crawl-log').replaceChildren();
      document.getElementById('crawl-counts').textContent = '0 trang';
      document.getElementById('crawl-stage').textContent = 'Đang chuẩn bị…';
      setCrawlRunning(true);
      crawlController = new AbortController();
      try {
        const response = await fetch('/api/learn/site', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
          signal: crawlController.signal,
          body: JSON.stringify({
            url,
            topic: document.getElementById('crawl-topic').value.trim(),
            depth: 10,
            max_pages: Number(document.getElementById('crawl-max-pages').value) || 300,
          }),
        });
        if (!response.ok) {
          let message = `HTTP ${response.status}`;
          try { message = apiErrorMessage(await response.json(), message); } catch (_) {}
          throw new Error(message);
        }
        await readLearnSSE(response, consumeCrawlEvent);
      } catch (error) {
        if (error.name === 'AbortError') {
          crawlLog('Đã dừng theo yêu cầu. Những trang đã học vẫn được giữ lại.', 'warn');
        } else {
          crawlLog(error.message || 'Duyệt website thất bại.', 'error');
          showToast(`Duyệt website thất bại: ${error.message || 'lỗi không xác định'}`, 'error');
        }
      } finally {
        crawlController = null;
        setCrawlRunning(false);
      }
    }

    function stopSiteCrawl() {
      if (crawlController) crawlController.abort();
    }

    async function startDeepLearn() {
      if (learnRunning) return;
      const topic = document.getElementById('deep-topic').value.trim();
      if (topic.length < 2) { showToast('Nhập chủ đề ít nhất 2 ký tự', 'error'); return; }
      const button = document.getElementById('btn-deep-learn');
      learnRunning = true;
      button.disabled = true; button.classList.add('loading');
      resetLearnDashboard(); setLearnState('running');
      document.getElementById('learn-excerpt').textContent = `Khởi động máy tự học cho chủ đề: ${topic}`;
      logLearn(`Bắt đầu tự học: ${topic}`, 'info');
      try {
        const response = await fetch('/api/learn/deep', {
          method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
          body: JSON.stringify({
            topic,
            keywords: document.getElementById('deep-keywords').value.trim(),
            depth: Number(document.getElementById('deep-depth').value),
            max_items: Number(document.getElementById('deep-items').value),
          }),
        });
        if (!response.ok) {
          let message = `HTTP ${response.status}`;
          try { message = apiErrorMessage(await response.json(), message); } catch (_) {}
          throw new Error(message);
        }
        await readLearnSSE(response);
      } catch (error) {
        setLearnState('error');
        logLearn(error.message || 'Không thể bắt đầu phiên tự học.', 'error');
        showToast(`Tự học thất bại: ${error.message || 'lỗi không xác định'}`, 'error');
      } finally {
        learnRunning = false;
        button.disabled = false; button.classList.remove('loading');
      }
    }

    function formatLearnDate(value) {
      if (!value) return '—';
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'short' });
    }

    async function loadLearnHistory() {
      const list = document.getElementById('learn-history');
      if (!list) return;
      list.textContent = 'Đang tải lịch sử…'; list.className = 'space-y-2 max-h-[280px] overflow-y-auto pr-1 text-xs text-zinc-500';
      try {
        const response = await fetch('/api/learn/runs?limit=20');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const items = Array.isArray(data) ? data : (data.items || []);
        list.replaceChildren(); list.className = 'space-y-2 max-h-[280px] overflow-y-auto pr-1';
        if (!items.length) {
          const empty = document.createElement('div'); empty.className = 'text-xs text-zinc-600 py-2'; empty.textContent = 'Chưa có phiên tự học nào.'; list.appendChild(empty); return;
        }
        for (const item of items) {
          const card = document.createElement('button');
          card.type = 'button'; card.className = 'learn-hist w-full text-left rounded-xl border border-zinc-800 px-3 py-2.5'; card.dataset.runId = item.id;
          const top = document.createElement('div'); top.className = 'flex items-center justify-between gap-2';
          const topic = document.createElement('span'); topic.className = 'truncate text-xs text-zinc-200'; topic.textContent = item.topic || 'Không rõ chủ đề';
          const state = document.createElement('span'); state.className = `learn-tag ${item.status === 'done' ? 'success' : item.status === 'error' ? 'error' : 'queued'}`; state.textContent = item.status || 'pending';
          top.append(topic, state);
          const meta = document.createElement('div'); meta.className = 'text-[10px] text-zinc-600 mt-1';
          const summary = item.summary || {};
          meta.textContent = `${formatLearnDate(item.created_at)} · ${summary.pages || 0} trang · ${summary.ingested || 0} đã nạp`;
          card.append(top, meta); card.addEventListener('click', () => showLearnRun(item.id, card)); list.appendChild(card);
        }
      } catch (error) {
        list.className = 'space-y-2 max-h-[280px] overflow-y-auto pr-1 text-xs text-red-400'; list.textContent = `Không tải được lịch sử: ${error.message}`;
      }
    }

    async function showLearnRun(runId, card) {
      document.querySelectorAll('#learn-history .learn-hist').forEach(el => el.classList.remove('active'));
      if (card) card.classList.add('active');
      const detail = document.getElementById('learn-history-detail');
      detail.className = 'mt-3 rounded-xl border border-zinc-800 bg-black/20 p-3 text-xs text-zinc-500'; detail.textContent = 'Đang tải chi tiết…';
      try {
        const response = await fetch(`/api/learn/runs/${encodeURIComponent(runId)}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const run = data.run || {}; const urls = data.urls || [];
        detail.replaceChildren(); detail.className = 'mt-3 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-3 text-xs';
        const heading = document.createElement('div'); heading.className = 'text-emerald-300 font-medium truncate'; heading.textContent = run.topic || 'Phiên tự học';
        const stats = document.createElement('div'); stats.className = 'text-zinc-400 mt-1';
        stats.textContent = `${urls.length} URL audit · ${formatLearnDate(run.created_at)}`;
        const logs = document.createElement('div'); logs.className = 'mt-2 max-h-28 overflow-y-auto font-mono text-[10px] leading-relaxed text-zinc-500';
        const entries = Array.isArray(run.log) ? run.log.slice(-8) : [];
        logs.textContent = entries.map(entry => entry.text || String(entry)).join('\n') || 'Không có log đã lưu.';
        detail.append(heading, stats, logs);
      } catch (error) {
        detail.className = 'mt-3 rounded-xl border border-red-500/20 bg-red-500/5 p-3 text-xs text-red-300'; detail.textContent = `Không tải được chi tiết: ${error.message}`;
      }
    }

    // ========== UTILS ==========
    function escapeHtml(str) {
      if (!str) return '';
      return str.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
    }

    // Backend errors use {"error": {"code", "message"}} — never FastAPI's "detail".
    function apiErrorMessage(payload, fallback) {
      return payload?.error?.message || fallback;
    }

    function renderAgentTrace(steps, toolCalls) {
      let html = `<div class="ml-0 mt-1 mb-3"><div class="text-[11px] text-zinc-500 mb-2">🧩 Agent trace</div>`;
      steps.forEach((s, i) => {
        const name = s.tool || (toolCalls[i]?.function?.name) || 'tool';
        const args = s.args || toolCalls[i]?.function?.arguments || {};
        const argsStr = typeof args === 'string' ? args : JSON.stringify(args);
        html += `<div class="agent-step inline-flex items-center gap-1 text-[10px] text-emerald-400 border border-emerald-900/50 mr-1 mb-1"><i class="fa-solid fa-wrench mr-1"></i>${escapeHtml(name)}</div>`;
      });
      html += `</div>`;
      return html;
    }

    async function showStats() {
      try {
        const res = await fetch('/api/memory/stats');
        const data = await res.json();
        alert(`Braind Stats\nSources: ${data.sources}\nDocuments: ${data.documents}\nChunks: ${data.chunks}\nGenerated: ${data.generated}\nLinks: ${data.links}`);
      } catch (e) {
        showToast('Không lấy được thống kê', 'error');
      }
    }

    async function syncVault() {
      showToast('Đang đồng bộ vault...', 'info');
      try {
        const res = await fetch('/api/ingest/vault', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(apiErrorMessage(data, 'Đồng bộ vault thất bại'));
        showToast(`Vault: +${data.added || 0} / ~${data.updated || 0} / -${data.deleted || 0}`);
      } catch (e) {
        showToast(e.message, 'error');
      }
    }

    // ========== NEURAL INGESTOR (canvas + pipeline) ==========
    let neuralRAF = null;
    let neuralStarted = false;
    const neuralNodes = [];
    const neuralConnections = [];
    let neuralActivity = 0; // 0..1 burst intensity on ingest

    function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

    function initNeuralIngestor() {
      const c = document.getElementById('neural-canvas');
      if (!c || neuralStarted) return;
      neuralStarted = true;
      const ctx = c.getContext('2d');
      function size() {
        const dpr = window.devicePixelRatio || 1;
        const w = c.clientWidth || 320, h = c.clientHeight || 420;
        c.width = Math.max(200, Math.round(w * dpr));
        c.height = Math.max(200, Math.round(h * dpr));
      }
      size();
      window.addEventListener('resize', size);
      // neural field
      const cols = 14, rows = 9;
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const n = {
            x: (x + 0.5) / cols, y: (y + 0.5) / rows,
            seed: Math.random() * Math.PI * 2,
            energy: 0
          };
          neuralNodes.push(n);
          // connect neighbors right + down
          if (x < cols - 1) neuralConnections.push([n, neuralNodes[neuralNodes.length - 1] && { x: (x + 1.5) / cols, y: (y + 0.5) / rows }]);
          if (y < rows - 1) neuralConnections.push([n, { x: (x + 0.5) / cols, y: (y + 1.5) / rows }]);
        }
      }
      // fix connection refs: we built pseudo-nodes above; rebuild properly
      rebuildNeuralGrid(cols, rows);

      let lastT = performance.now();
      function frame(t) {
        const dt = (t - lastT) / 1000; lastT = t;
        ctx.clearRect(0, 0, c.width, c.height);
        const w = c.width, h = c.height;
        // pulse origin (center-right brain core)
        const coreX = w * 0.82, coreY = h * 0.4;
        ctx.save();
        // connections
        ctx.lineWidth = 1;
        for (const [a, b] of neuralConnections) {
          const ax = a.x * w, ay = a.y * h, bx = b.x * w, by = b.y * h;
          const actA = a.energy, actB = b.energy;
          const alpha = 0.06 + (actA + actB) * 0.5;
          ctx.strokeStyle = `rgba(34,211,238,${Math.min(0.5, alpha)})`;
          ctx.beginPath();
          const mx = (ax + bx) / 2, my = (ay + by) / 2;
          ctx.moveTo(ax, ay);
          ctx.quadraticCurveTo(mx + (Math.sin(t * 0.004 + a.seed) * 6), my, bx, by);
          ctx.stroke();
        }
        // nodes
        for (const n of neuralNodes) {
          const x = n.x * w, y = n.y * h;
          const idle = 0.35 + 0.15 * Math.sin(t * 0.002 + n.seed);
          const r = (1.5 + n.energy * 3.5) * (0.9 + 0.1 * idle);
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${neuralActivity > 0.05 ? '34,211,238' : '148,163,184'},${Math.min(0.9, idle + n.energy)})`;
          ctx.fill();
          // decay
          n.energy *= 0.95;
        }
        // activity bursts feeding from left (ingest source) to core
        if (neuralActivity > 0.02) {
          const streamY = h * 0.4;
          ctx.strokeStyle = 'rgba(34,211,238,0.35)';
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 10]);
          ctx.lineDashOffset = -t * 0.03;
          ctx.beginPath();
          ctx.moveTo(0, streamY);
          ctx.quadraticCurveTo(w * 0.4, streamY - 40, coreX, coreY);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        // core glow
        const corePulse = 0.5 + 0.5 * Math.sin(t * 0.003);
        const grad = ctx.createRadialGradient(coreX, coreY, 2, coreX, coreY, 46 + neuralActivity * 40);
        grad.addColorStop(0, `rgba(34,211,238,${0.35 + neuralActivity * 0.4 * corePulse})`);
        grad.addColorStop(1, 'rgba(34,211,238,0)');
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(coreX, coreY, 46 + neuralActivity * 40, 0, Math.PI * 2); ctx.fill();
        // core ring
        ctx.strokeStyle = `rgba(167,139,250,${0.35 + neuralActivity * 0.4})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(coreX, coreY, 20 + neuralActivity * 18 + Math.sin(t * 0.004) * 3, 0, Math.PI * 2); ctx.stroke();
        neuralActivity *= 0.96;
        ctx.restore();
        neuralRAF = requestAnimationFrame(frame);
      }
      neuralRAF = requestAnimationFrame(frame);
    }

    function rebuildNeuralGrid(cols, rows) {
      neuralNodes.length = 0;
      neuralConnections.length = 0;
      const grid = [];
      for (let y = 0; y < rows; y++) {
        grid[y] = [];
        for (let x = 0; x < cols; x++) {
          const n = { x: (x + 0.5) / cols, y: (y + 0.5) / rows, seed: Math.random() * Math.PI * 2, energy: 0 };
          grid[y][x] = n;
          neuralNodes.push(n);
        }
      }
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          if (x < cols - 1) neuralConnections.push([grid[y][x], grid[y][x + 1]]);
          if (y < rows - 1) neuralConnections.push([grid[y][x], grid[y + 1][x]]);
        }
      }
    }

    const PIPE_LABELS = ['Nhận', 'Chunk', 'Embed', 'Từ khóa', 'Liên kết'];

    function setPipeStep(i, state, { resetOthers = true } = {}) {
      if (resetOthers) {
        document.querySelectorAll('#neural-pipeline .pipe-step').forEach(el => {
          el.classList.remove('active', 'done', 'fail');
          el.classList.add('idle');
        });
      }
      const el = document.querySelector(`#neural-pipeline .pipe-step[data-step="${i}"]`);
      if (el) {
        el.classList.remove('idle');
        el.classList.add(state);
        if (state === 'active') el.scrollIntoView({ block: 'nearest' });
      }
    }
    function setPipeStepAll(state) {
      document.querySelectorAll('#neural-pipeline .pipe-step').forEach(el => {
        el.classList.remove('active', 'done', 'fail', 'idle');
        el.classList.add(state);
      });
    }

    function setNeuralState(mode, title, sub) {
      const el = document.getElementById('neural-state');
      const t = document.getElementById('neural-status-title');
      const s = document.getElementById('neural-status-sub');
      if (el) {
        if (mode === 'done') { el.textContent = 'DONE'; el.style.color = '#34d399'; }
        else if (mode === 'error') { el.textContent = 'ERROR'; el.style.color = '#f87171'; }
        else { el.textContent = 'RUNNING'; el.style.color = '#22d3ee'; }
      }
      if (t) t.textContent = title || 'Đang xử lý';
      if (s) s.textContent = sub || '';
    }

    function updateNeuralProgress(pct) {
      const bar = document.getElementById('neural-progress');
      if (bar) bar.style.width = Math.max(0, Math.min(100, pct)) + '%';
    }

    function logIngest(msg, cls) {
      const log = document.getElementById('ingest-log');
      const body = document.getElementById('ingest-log-body');
      if (!log || !body) return;
      log.style.display = '';
      const line = document.createElement('div');
      line.className = cls || 'info';
      const ts = new Date().toLocaleTimeString('vi-VN', { hour12: false });
      line.textContent = `[${ts}] ${msg}`;
      body.appendChild(line);
      body.scrollTop = body.scrollHeight;
    }

    function clearIngestLog() {
      const body = document.getElementById('ingest-log-body');
      if (body) body.innerHTML = '';
    }

    // Idle neural bumps + burst on ingest
    setInterval(() => {
      if (neuralNodes.length) {
        const n = neuralNodes[(Math.random() * neuralNodes.length) | 0];
        n.energy = Math.min(1, n.energy + 0.6);
      }
    }, 700);

    function startIngestPipeline() {
      const shell = document.getElementById('ingest-shell');
      if (shell) shell.classList.add('ingesting');
      const btn = document.getElementById('ingest-btn');
      if (btn) btn.classList.add('loading');
      const input = document.getElementById('ingest-text');
      if (input) input.disabled = true;
      document.getElementById('ingest-title') && (document.getElementById('ingest-title').disabled = true);
      updateNeuralProgress(6);
      burstNeural();
      setNeuralState('run', 'Đang xử lý', 'Đợi...');
      logIngest('Pipeline khởi động.', 'info');
      logIngest('Đang gửi dữ liệu tới braind core...', 'info');
    }

    function stopIngestPipeline() {
      const shell = document.getElementById('ingest-shell');
      if (shell) shell.classList.remove('ingesting');
      const btn = document.getElementById('ingest-btn');
      if (btn) btn.classList.remove('loading');
      const input = document.getElementById('ingest-text');
      if (input) input.disabled = false;
      document.getElementById('ingest-title') && (document.getElementById('ingest-title').disabled = false);
    }

    function burstNeural(power) {
      neuralActivity = Math.max(neuralActivity, power || 0.8);
      // energize a chain of nodes toward the core
      for (const n of neuralNodes) {
        if (Math.random() < 0.28) n.energy = Math.min(1, n.energy + (power || 0.8) * 0.7);
      }
    }
    setInterval(() => {
      if (!document.getElementById('view-ingest').classList.contains('hidden')) {
        if (neuralActivity < 0.05) neuralActivity = 0.12;
      }
    }, 6000);

    // Dom drag-drop wiring for the new dropzone
    (function wireDrop() {
      const dz = document.getElementById('ingest-dropzone');
      const fi = document.getElementById('file-input');
      if (dz) {
        dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
        dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
        dz.addEventListener('drop', e => {
          e.preventDefault(); dz.classList.remove('dragover');
          if (e.dataTransfer.files && e.dataTransfer.files[0] && fi) {
            fi.files = e.dataTransfer.files;
            handleFileUpload(fi);
          }
        });
      }
    })();

    // Keyboard + init
    document.addEventListener('DOMContentLoaded', async () => {
      initializeTailwind();
      await loadNav();
      await loadCommands();
      loadWriterOptions();
      const input = document.getElementById('chat-input');
      if (input) {
        input.addEventListener('keydown', e => {
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        });
      }
      // Command palette: Ctrl+K / Cmd+K
      document.addEventListener('keydown', e => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
          e.preventDefault();
          openCommandPalette();
        }
      });
      // Default view
      switchView('chat');
      // Warm up memory graph in background
      setTimeout(() => { if (!document.getElementById('view-memory').classList.contains('hidden')) loadMemoryGraph(); }, 800);
    });
