/**
 * SieveRAG Command Center Application Logic
 * Interactive Temporal Video RAG, Click-to-Seek Video Streaming & Multi-modal Telemetry
 */

(() => {
  "use strict";

  // Application State
  const state = {
    activeVideoId: null,
    activeVideoData: null,
    activeEpisodeIndex: null,
    videoList: [],
    stats: null,
  };

  // DOM Elements Cache
  const el = {
    // Telemetry Pills
    statVideos: document.getElementById("stat-videos"),
    statStoryboards: document.getElementById("stat-storyboards"),
    statDisturbances: document.getElementById("stat-disturbances"),
    statSavings: document.getElementById("stat-savings"),
    statQwenKv: document.getElementById("stat-qwen-kv"),
    statBackend: document.getElementById("stat-backend"),
    btnRefreshStats: document.getElementById("btn-refresh-stats"),

    // Ingestion Form
    ingestForm: document.getElementById("ingest-form"),
    tabFileMode: document.getElementById("tab-file-mode"),
    tabLocalMode: document.getElementById("tab-local-mode"),
    contentFileMode: document.getElementById("content-file-mode"),
    contentLocalMode: document.getElementById("content-local-mode"),
    dropZone: document.getElementById("drop-zone"),
    videoFileInput: document.getElementById("video-file-input"),
    fileLabel: document.getElementById("file-label"),
    localPathInput: document.getElementById("local-path-input"),
    targetVlmSelect: document.getElementById("target-vlm-select"),
    runVlmCheck: document.getElementById("run-vlm-check"),
    indexEpisodesCheck: document.getElementById("index-episodes-check"),
    btnIngest: document.getElementById("btn-ingest"),
    ingestSpinner: document.getElementById("ingest-spinner"),

    // Video Library
    libraryCount: document.getElementById("library-count"),
    btnRefreshLibrary: document.getElementById("btn-refresh-library"),
    videoCardsContainer: document.getElementById("video-cards-container"),

    // Center Viewer
    activeVideoTitle: document.getElementById("active-video-title"),
    activeVideoMeta: document.getElementById("active-video-meta"),
    playerTimeDisplay: document.getElementById("player-time-display"),
    mainVideoPlayer: document.getElementById("main-video-player"),
    playerPlaceholder: document.getElementById("player-placeholder"),

    // Timeline Scrubber
    timelineTrackContainer: document.getElementById("timeline-track-container"),
    timelineTrack: document.getElementById("timeline-track"),
    timelinePlayhead: document.getElementById("timeline-playhead"),

    // Episode Inspector
    inspectorEpTitle: document.getElementById("inspector-ep-title"),
    inspectorEpBadge: document.getElementById("inspector-ep-badge"),
    inspectorEpTime: document.getElementById("inspector-ep-time"),
    inspectorStoryboardImg: document.getElementById("inspector-storyboard-img"),
    storyboardPlaceholder: document.getElementById("storyboard-placeholder"),
    btnZoomStoryboard: document.getElementById("btn-zoom-storyboard"),
    inspectorAudioPlayer: document.getElementById("inspector-audio-player"),
    inspectorSummary: document.getElementById("inspector-summary"),
    inspectorActors: document.getElementById("inspector-actors"),
    inspectorObjects: document.getElementById("inspector-objects"),
    inspectorTranscript: document.getElementById("inspector-transcript"),
    inspectorGridSpec: document.getElementById("inspector-grid-spec"),
    inspectorTileActions: document.getElementById("inspector-tile-actions"),

    // RAG Console
    ragQueryInput: document.getElementById("rag-query-input"),
    btnToggleFilters: document.getElementById("btn-toggle-filters"),
    ragFiltersPanel: document.getElementById("rag-filters-panel"),
    ragFilterVideo: document.getElementById("rag-filter-video"),
    ragFilterSeverity: document.getElementById("rag-filter-severity"),
    ragFilterTopk: document.getElementById("rag-filter-topk"),
    ragFilterAnomaly: document.getElementById("rag-filter-anomaly"),
    ragFilterStoryboards: document.getElementById("rag-filter-storyboards"),
    btnRunSearch: document.getElementById("btn-run-search"),
    btnRunAsk: document.getElementById("btn-run-ask"),
    ragResultsContainer: document.getElementById("rag-results-container"),

    // Modal
    storyboardModal: document.getElementById("storyboard-modal"),
    modalBackdrop: document.getElementById("modal-backdrop"),
    modalStoryboardTitle: document.getElementById("modal-storyboard-title"),
    modalStoryboardImg: document.getElementById("modal-storyboard-img"),
    btnCloseModal: document.getElementById("btn-close-modal"),

    // Toasts
    toastContainer: document.getElementById("toast-container"),
  };

  /* ==========================================================================
     Formatters & Utilities
     ========================================================================== */

  function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return "00:00.00";
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    const mStr = String(mins).padStart(2, "0");
    const sStr = secs.toFixed(2).padStart(5, "0");
    return `${mStr}:${sStr}`;
  }

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    el.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      toast.style.transition = "all 0.3s";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  /* ==========================================================================
     Telemetry & Global Stats
     ========================================================================== */

  async function fetchStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      state.stats = data;

      el.statVideos.textContent = data.total_videos || 0;
      el.statStoryboards.textContent = data.total_storyboards_emitted || 0;
      el.statDisturbances.textContent = data.total_disturbances_suppressed || 0;
      el.statSavings.textContent = `${(data.overall_tokens_saved_pct || 0).toFixed(1)}%`;
      el.statQwenKv.textContent = `${(data.total_qwen7b_kv_cache_mb || 0).toFixed(1)} MB`;
      el.statBackend.textContent = (data.vector_backend || "local").toUpperCase();
    } catch (err) {
      console.warn("Failed to fetch global stats:", err);
    }
  }

  /* ==========================================================================
     Video Library & Ingestion
     ========================================================================== */

  async function fetchVideos() {
    try {
      const res = await fetch("/api/videos");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const list = await res.json();
      state.videoList = list;

      el.libraryCount.textContent = list.length;
      updateVideoFilterDropdown(list);

      if (list.length === 0) {
        el.videoCardsContainer.innerHTML = `
          <div class="empty-state">No videos ingested yet. Upload or ingest a video above.</div>
        `;
        return;
      }

      el.videoCardsContainer.innerHTML = "";
      list.forEach((vid) => {
        const card = document.createElement("div");
        card.className = `video-card ${state.activeVideoId === vid.video_id ? "active" : ""}`;
        card.dataset.videoId = vid.video_id;

        card.innerHTML = `
          <div class="video-card-top">
            <span class="video-card-name" title="${vid.video_filename}">${vid.video_filename}</span>
            <button class="btn-card-del" title="Delete Video" data-delete-id="${vid.video_id}">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 6h18"></path>
                <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
                <line x1="10" y1="11" x2="10" y2="17"></line>
                <line x1="14" y1="11" x2="14" y2="17"></line>
              </svg>
            </button>
          </div>
          <div class="video-card-stats">
            <span>${formatTime(vid.duration_sec)} &bull; ${vid.storyboards_emitted} SBs</span>
            <span class="card-pill-savings">${(vid.tokens_saved_pct || 0).toFixed(1)}% Saved</span>
          </div>
        `;

        card.addEventListener("click", (e) => {
          if (e.target.closest(".btn-card-del")) return;
          loadVideo(vid.video_id);
        });

        const delBtn = card.querySelector(".btn-card-del");
        delBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          deleteVideo(vid.video_id, vid.video_filename);
        });

        el.videoCardsContainer.appendChild(card);
      });
    } catch (err) {
      console.error("Failed to load video library:", err);
      showToast("Failed to load video corpus library.", "error");
    }
  }

  function updateVideoFilterDropdown(list) {
    const currentVal = el.ragFilterVideo.value;
    el.ragFilterVideo.innerHTML = `<option value="">All Videos in Corpus</option>`;
    list.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v.video_id;
      opt.textContent = `${v.video_filename} (${v.video_id.slice(0, 8)})`;
      el.ragFilterVideo.appendChild(opt);
    });
    if (list.some((v) => v.video_id === currentVal)) {
      el.ragFilterVideo.value = currentVal;
    }
  }

  async function deleteVideo(videoId, filename) {
    if (!confirm(`Are you sure you want to delete video '${filename}' and all its indexed vectors?`)) {
      return;
    }

    try {
      const res = await fetch(`/api/videos/${videoId}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      showToast(`Deleted video (${data.removed_vectors || 0} vectors removed).`, "success");

      if (state.activeVideoId === videoId) {
        resetViewer();
      }

      await fetchStats();
      await fetchVideos();
    } catch (err) {
      console.error("Failed to delete video:", err);
      showToast("Error deleting video.", "error");
    }
  }

  function resetViewer() {
    state.activeVideoId = null;
    state.activeVideoData = null;
    state.activeEpisodeIndex = null;

    el.mainVideoPlayer.pause();
    el.mainVideoPlayer.removeAttribute("src");
    el.mainVideoPlayer.load();

    el.activeVideoTitle.textContent = "No Video Selected";
    el.activeVideoMeta.className = "badge badge-info hidden";
    el.playerPlaceholder.classList.remove("hidden");
    el.timelineTrack.innerHTML = '<div class="playhead" id="timeline-playhead"></div>';
    resetInspector();
  }

  function resetInspector() {
    el.inspectorEpTitle.textContent = "Select an Episode";
    el.inspectorEpBadge.className = "badge hidden";
    el.inspectorEpTime.textContent = "--";
    el.inspectorStoryboardImg.src = "";
    el.inspectorStoryboardImg.classList.add("hidden");
    el.storyboardPlaceholder.classList.remove("hidden");
    el.btnZoomStoryboard.classList.add("hidden");
    el.inspectorAudioPlayer.pause();
    el.inspectorAudioPlayer.removeAttribute("src");
    el.inspectorSummary.textContent = "Click an episode segment on the timeline above to view full multimodal telemetry.";
    el.inspectorActors.innerHTML = "&mdash;";
    el.inspectorObjects.innerHTML = "&mdash;";
    el.inspectorTranscript.innerHTML = "&mdash;";
    el.inspectorGridSpec.textContent = "Grid: --";
    el.inspectorTileActions.innerHTML = '<div class="empty-tile-prompt">No tile actions detected.</div>';
  }

  /* ==========================================================================
     Ingestion Form Submission
     ========================================================================== */

  async function handleIngestSubmit(e) {
    e.preventDefault();

    const isFileMode = el.tabFileMode.classList.contains("active");
    const file = el.videoFileInput.files[0];
    const localPath = el.localPathInput.value.trim();

    if (isFileMode && !file) {
      showToast("Please choose a video file to upload.", "error");
      return;
    }
    if (!isFileMode && !localPath) {
      showToast("Please enter a local video file path.", "error");
      return;
    }

    const formData = new FormData();
    if (isFileMode && file) {
      formData.append("file", file);
    } else if (localPath) {
      formData.append("local_video_path", localPath);
    }

    formData.append("target_vlm", el.targetVlmSelect.value);
    formData.append("run_vlm_extraction", el.runVlmCheck.checked);
    formData.append("index_episodes", el.indexEpisodesCheck.checked);

    // UI state: submitting
    el.btnIngest.disabled = true;
    el.ingestSpinner.classList.remove("hidden");
    el.btnIngest.querySelector(".btn-text").textContent = "Sieving & Ingesting...";

    try {
      const res = await fetch("/api/videos/ingest", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${res.status}`);
      }

      const receipt = await res.json();
      showToast(
        `Video ingested! Emitted ${receipt.storyboards_emitted} storyboards (${receipt.tokens_saved_pct}% tokens saved).`,
        "success"
      );

      // Reset form input
      el.videoFileInput.value = "";
      el.localPathInput.value = "";
      el.fileLabel.textContent = "Choose video or drag here";

      await fetchStats();
      await fetchVideos();

      // Automatically load the newly ingested video
      if (receipt.video_id) {
        await loadVideo(receipt.video_id);
      }
    } catch (err) {
      console.error("Ingestion failed:", err);
      showToast(`Ingestion failed: ${err.message}`, "error");
    } finally {
      el.btnIngest.disabled = false;
      el.ingestSpinner.classList.add("hidden");
      el.btnIngest.querySelector(".btn-text").textContent = "Ingest & Sieve Video";
    }
  }

  /* ==========================================================================
     Viewer & Timeline Scrubber
     ========================================================================== */

  async function loadVideo(videoId) {
    try {
      const res = await fetch(`/api/videos/${videoId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      state.activeVideoId = videoId;

    const scopeSelect = document.getElementById("filter-video-scope") || document.querySelector("select[id*='video'], select[id*='scope']");
    if (scopeSelect && videoId) {
        scopeSelect.value = videoId;
    }

      state.activeVideoData = data;
      state.activeEpisodeIndex = null;

      // Update Library Card active states
      document.querySelectorAll(".video-card").forEach((c) => {
        c.classList.toggle("active", c.dataset.videoId === videoId);
      });

      // Update Header info
      el.activeVideoTitle.textContent = data.video_filename || videoId;
      el.activeVideoMeta.textContent = `${formatTime(data.duration_sec)} • ${data.episodes ? data.episodes.length : 0} Episodes • ${data.tokens_saved_pct || 0}% Saved`;
      el.activeVideoMeta.className = "badge badge-info";

      // Load Video Player
      if (data.video_url) {
        el.mainVideoPlayer.src = data.video_url;
        el.playerPlaceholder.classList.add("hidden");
      } else {
        el.mainVideoPlayer.removeAttribute("src");
        el.playerPlaceholder.classList.remove("hidden");
        el.playerPlaceholder.textContent = "Video file not directly streamable.";
      }

      // Build Interactive Timeline
      renderTimeline(data);

      // Select first episode if present
      if (data.episodes && data.episodes.length > 0) {
        selectEpisode(data.episodes[0].episode_index);
      } else {
        resetInspector();
        el.inspectorSummary.textContent = "No motion episodes detected in this video.";
      }
    } catch (err) {
      console.error("Failed to load video details:", err);
      showToast("Error loading video details.", "error");
    }
  }

  function renderTimeline(videoData) {
    const duration = parseFloat(videoData.duration_sec) || 1.0;
    el.timelineTrack.innerHTML = '<div class="playhead" id="timeline-playhead"></div>';

    const episodes = videoData.episodes || [];
    episodes.forEach((ep) => {
      const seg = document.createElement("div");
      const isAnomaly = ep.anomaly_or_security_flag || ep.severity === "high";
      seg.className = `timeline-segment ${isAnomaly ? "anomaly-segment" : ""}`;
      seg.dataset.episodeIndex = ep.episode_index;

      const leftPct = Math.max(0, (ep.start_time_sec / duration) * 100);
      const widthPct = Math.max(1.0, (ep.duration_sec / duration) * 100);

      seg.style.left = `${leftPct}%`;
      seg.style.width = `${widthPct}%`;
      seg.title = `Ep #${ep.episode_index}: ${ep.episode_title || "Motion"} (${ep.start_time_sec.toFixed(1)}s - ${ep.end_time_sec.toFixed(1)}s)`;

      seg.addEventListener("click", (e) => {
        e.stopPropagation();
        selectEpisode(ep.episode_index);
        seekVideo(ep.start_time_sec);
      });

      el.timelineTrack.appendChild(seg);
    });
  }

  function seekVideo(timeSec) {
    if (!el.mainVideoPlayer.src) return;
    el.mainVideoPlayer.currentTime = Math.max(0, timeSec);
    el.mainVideoPlayer.play().catch(() => {});
  }

  /* ==========================================================================
     Episode Inspector & Clickable Tile Actions
     ========================================================================== */

  function selectEpisode(episodeIndex) {
    if (!state.activeVideoData || !state.activeVideoData.episodes) return;
    const ep = state.activeVideoData.episodes.find((e) => e.episode_index === episodeIndex);
    if (!ep) return;

    state.activeEpisodeIndex = episodeIndex;

    // Highlight timeline segment
    document.querySelectorAll(".timeline-segment").forEach((s) => {
      s.classList.toggle("active-segment", parseInt(s.dataset.episodeIndex, 10) === episodeIndex);
    });

    // Inspector Header
    el.inspectorEpTitle.textContent = ep.episode_title || `Motion Episode #${ep.episode_index}`;
    if (ep.anomaly_or_security_flag) {
      el.inspectorEpBadge.textContent = `ALERT: ${ep.severity.toUpperCase()}`;
      el.inspectorEpBadge.className = "badge badge-anomaly";
    } else {
      el.inspectorEpBadge.textContent = `${ep.severity.toUpperCase()} SEVERITY`;
      el.inspectorEpBadge.className = "badge badge-info";
    }
    el.inspectorEpTime.textContent = `${formatTime(ep.start_time_sec)} - ${formatTime(ep.end_time_sec)} (${ep.duration_sec.toFixed(2)}s)`;

    // Storyboard
    if (ep.storyboard_url) {
      el.inspectorStoryboardImg.src = ep.storyboard_url;
      el.inspectorStoryboardImg.classList.remove("hidden");
      el.storyboardPlaceholder.classList.add("hidden");
      el.btnZoomStoryboard.classList.remove("hidden");
    } else {
      el.inspectorStoryboardImg.src = "";
      el.inspectorStoryboardImg.classList.add("hidden");
      el.storyboardPlaceholder.classList.remove("hidden");
      el.btnZoomStoryboard.classList.add("hidden");
    }

    // Audio Slice Player
    if (ep.audio_url) {
      el.inspectorAudioPlayer.src = ep.audio_url;
      el.inspectorAudioPlayer.parentElement.classList.remove("hidden");
    } else {
      el.inspectorAudioPlayer.pause();
      el.inspectorAudioPlayer.removeAttribute("src");
      el.inspectorAudioPlayer.parentElement.classList.add("hidden");
    }

    // Textual Telemetry
    el.inspectorSummary.textContent = ep.narrative_summary || "No narrative summary generated.";
    
    // Actors
    el.inspectorActors.innerHTML = "";
    if (ep.actors && ep.actors.length > 0) {
      ep.actors.forEach((a) => {
        const p = document.createElement("span");
        p.className = "tag-pill";
        p.textContent = a;
        el.inspectorActors.appendChild(p);
      });
    } else {
      el.inspectorActors.innerHTML = "&mdash;";
    }

    // Objects
    el.inspectorObjects.innerHTML = "";
    if (ep.objects_detected && ep.objects_detected.length > 0) {
      ep.objects_detected.forEach((o) => {
        const p = document.createElement("span");
        p.className = "tag-pill";
        p.textContent = o;
        el.inspectorObjects.appendChild(p);
      });
    } else {
      el.inspectorObjects.innerHTML = "&mdash;";
    }

    // Transcript / Audio Cues
    el.inspectorTranscript.textContent = ep.audio_transcript_and_cues || "No significant audio events transcribed.";

    // Grid Spec
    const gridStr = Array.isArray(ep.grid_used) ? `${ep.grid_used[0]}x${ep.grid_used[1]}` : (ep.grid_used || "2x3");
    el.inspectorGridSpec.textContent = `Grid: ${gridStr}`;

    // Itemized Tile Actions (Click to Seek Video)
    renderTileActions(ep.tile_actions, ep);
  }

  function renderTileActions(tileActions, episode) {
    el.inspectorTileActions.innerHTML = "";
    if (!tileActions || tileActions.length === 0) {
      el.inspectorTileActions.innerHTML = '<div class="empty-tile-prompt">No discrete tile actions annotated.</div>';
      return;
    }

    tileActions.forEach((action) => {
      const card = document.createElement("div");
      card.className = "tile-action-card";
      const absTime = typeof action.absolute_time_sec === "number" ? action.absolute_time_sec : episode.start_time_sec;
      const gridText = action.grid_cells && action.grid_cells.length > 0 ? action.grid_cells.join(",") : "";

      card.innerHTML = `
        <div class="tile-action-left">
          <span class="tile-pill-num">T#${action.tile_number}</span>
          <span class="tile-pill-time">${absTime.toFixed(2)}s</span>
          ${gridText ? `<span class="tile-pill-grid">${gridText}</span>` : ""}
        </div>
        <div class="tile-action-desc" title="${action.action_description || ""}">
          ${action.action_description || "Visual motion activity"}
        </div>
        <div class="tile-action-seek-icon" title="Seek video to ${absTime.toFixed(2)}s">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="6 3 20 12 6 21 6 3"></polygon>
          </svg>
        </div>
      `;

      card.addEventListener("click", () => {
        seekVideo(absTime);
        showToast(`Seeked video to ${absTime.toFixed(2)}s (Tile #${action.tile_number})`, "info");
      });

      el.inspectorTileActions.appendChild(card);
    });
  }

  /* ==========================================================================
     Temporal Video RAG Console (Search & Synthesis)
     ========================================================================== */

  async function handleSearch() {
    const query = el.ragQueryInput.value.trim();
    if (!query) {
      showToast("Please enter a search query.", "error");
      return;
    }

    const payload = {
      query: query,
      top_k: parseInt(el.ragFilterTopk.value, 10) || 4,
      video_id: el.ragFilterVideo.value || null,
      min_severity: el.ragFilterSeverity.value || null,
      anomaly_only: el.ragFilterAnomaly.checked,
    };

    el.btnRunSearch.disabled = true;
    el.btnRunSearch.textContent = "Searching...";

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderSearchResults(data.results || [], query);
    } catch (err) {
      console.error("Vector search failed:", err);
      showToast("Search failed.", "error");
    } finally {
      el.btnRunSearch.disabled = false;
      el.btnRunSearch.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        Vector Search
      `;
    }
  }

  function renderSearchResults(hits, query) {
    if (hits.length === 0) {
      el.ragResultsContainer.innerHTML = `
        <div class="rag-empty-state">
          <p>No episodes matched query "<em>${query}</em>" under current filters.</p>
        </div>
      `;
      return;
    }

    el.ragResultsContainer.innerHTML = "";
    hits.forEach((hit) => {
      const card = document.createElement("div");
      card.className = "search-hit-card";
      const scorePct = (hit.score * 100).toFixed(1);

      card.innerHTML = `
        <div class="search-hit-header">
          <span class="search-hit-title">Ep #${hit.episode_index}: ${hit.episode_title}</span>
          <span class="search-hit-score">${scorePct}% Match</span>
        </div>
        <p class="search-hit-summary">${hit.narrative_summary}</p>
        <div class="search-hit-footer">
          <span class="search-hit-timing">${formatTime(hit.start_time_sec)} &mdash; ${formatTime(hit.end_time_sec)}</span>
          <button class="btn btn-secondary btn-sm" data-seek-time="${hit.start_time_sec}" data-video-id="${hit.video_id}" data-ep-idx="${hit.episode_index}">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
              <polygon points="6 3 20 12 6 21 6 3"></polygon>
            </svg>
            Seek &amp; Inspect
          </button>
        </div>
      `;

      const btnSeek = card.querySelector("button");
      btnSeek.addEventListener("click", async () => {
        await jumpToEpisodeAndSeek(hit.video_id, hit.episode_index, hit.start_time_sec);
      });

      el.ragResultsContainer.appendChild(card);
    });
  }

  async function handleAsk() {
    const question = el.ragQueryInput.value.trim();
    if (!question) {
      showToast("Please enter a question to ask.", "error");
      return;
    }

    const payload = {
      question: question,
      top_k: parseInt(el.ragFilterTopk.value, 10) || 4,
      video_id: el.ragFilterVideo.value || null,
      min_severity: el.ragFilterSeverity.value || null,
      anomaly_only: el.ragFilterAnomaly.checked,
      include_storyboards: el.ragFilterStoryboards.checked,
    };

    el.btnRunAsk.disabled = true;
    el.btnRunAsk.textContent = "Synthesizing...";

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const answer = await res.json();
      renderRAGAnswer(answer);
    } catch (err) {
      console.error("RAG synthesis failed:", err);
      showToast("Question answering synthesis failed.", "error");
    } finally {
      el.btnRunAsk.disabled = false;
      el.btnRunAsk.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
        </svg>
        Ask SieveRAG
      `;
    }
  }

  function renderRAGAnswer(resp) {
    el.ragResultsContainer.innerHTML = "";

    const card = document.createElement("div");
    card.className = "rag-answer-card";

    // Badges
    const confBadge = `<span class="badge badge-info">${resp.confidence.toUpperCase()} CONFIDENCE</span>`;
    const alertBadge = resp.security_alert_triggered
      ? `<span class="badge badge-anomaly">SECURITY ALERT</span>`
      : "";

    card.innerHTML = `
      <div class="rag-answer-header">
        <h4>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          SieveRAG Grounded Response
        </h4>
        <div class="rag-badges-row">${alertBadge} ${confBadge}</div>
      </div>
      <div class="rag-answer-body">${resp.answer}</div>
    `;

    // Attached Storyboard Thumbnails
    if (resp.storyboards_attached && resp.storyboards_attached.length > 0) {
      const thumbsSec = document.createElement("div");
      thumbsSec.className = "attached-storyboards-section";
      thumbsSec.innerHTML = `<label style="font-size: 0.65rem; font-weight:700; text-transform:uppercase; color: var(--text-muted);">Verified Storyboard Frames:</label>`;
      const thumbsRow = document.createElement("div");
      thumbsRow.className = "attached-storyboards-row";

      resp.storyboards_attached.forEach((url) => {
        const thumb = document.createElement("div");
        thumb.className = "attached-thumb-box";
        thumb.innerHTML = `<img src="${url}" alt="Storyboard" />`;
        thumb.addEventListener("click", () => openModal(url, "Attached Storyboard"));
        thumbsRow.appendChild(thumb);
      });

      thumbsSec.appendChild(thumbsRow);
      card.appendChild(thumbsSec);
    }

    // Citations with Click-to-Seek
    if (resp.citations && resp.citations.length > 0) {
      const citSec = document.createElement("div");
      citSec.className = "rag-citations-section";
      citSec.innerHTML = `<label>Temporal &amp; Spatial Citations (Click to Seek Video):</label>`;
      const chipsContainer = document.createElement("div");
      chipsContainer.className = "citation-chips-container";

      resp.citations.forEach((cit) => {
        const chip = document.createElement("div");
        chip.className = "citation-chip";

        const tileText = cit.tile_number ? `Tile #${cit.tile_number}` : "";
        const timeVal = typeof cit.timestamp_sec === "number" ? cit.timestamp_sec : cit.start_time_sec;
        const gridText = cit.grid_cells && cit.grid_cells.length > 0 ? `Grid ${cit.grid_cells.join(",")}` : "";

        const codeParts = [`Ep #${cit.episode_index}`];
        if (tileText) codeParts.push(tileText);
        if (timeVal !== undefined) codeParts.push(`${timeVal.toFixed(2)}s`);
        if (gridText) codeParts.push(gridText);

        chip.innerHTML = `
          <div class="citation-chip-top">
            <span class="citation-badge-code">[${codeParts.join(" &bull; ")}]</span>
            <span class="citation-seek-icon">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="6 3 20 12 6 21 6 3"></polygon>
              </svg>
              SEEK
            </span>
          </div>
          <div class="citation-evidence">${cit.evidence}</div>
        `;

        chip.addEventListener("click", async () => {
          await jumpToEpisodeAndSeek(cit.video_id, cit.episode_index, timeVal);
          showToast(`Jumped to Citation: Ep #${cit.episode_index} at ${timeVal.toFixed(2)}s`, "info");
        });

        chipsContainer.appendChild(chip);
      });

      citSec.appendChild(chipsContainer);
      card.appendChild(citSec);
    }

    el.ragResultsContainer.appendChild(card);
  }

  async function jumpToEpisodeAndSeek(videoId, episodeIndex, seekSec) {
    if (state.activeVideoId !== videoId) {
      await loadVideo(videoId);
    }
    if (typeof episodeIndex === "number") {
      selectEpisode(episodeIndex);
    }
    seekVideo(seekSec);
  }

  /* ==========================================================================
     Modal & Zoom Handling
     ========================================================================== */

  function openModal(imgSrc, title = "Storyboard Preview") {
    el.modalStoryboardTitle.textContent = title;
    el.modalStoryboardImg.src = imgSrc;
    el.storyboardModal.classList.remove("hidden");
  }

  function closeModal() {
    el.storyboardModal.classList.add("hidden");
    el.modalStoryboardImg.src = "";
  }

  /* ==========================================================================
     Event Listeners Setup
     ========================================================================== */

  function setupEventListeners() {
    // Refresh stats & library
    el.btnRefreshStats.addEventListener("click", fetchStats);
    el.btnRefreshLibrary.addEventListener("click", fetchVideos);

    // Ingestion tabs
    el.tabFileMode.addEventListener("click", () => {
      el.tabFileMode.classList.add("active");
      el.tabLocalMode.classList.remove("active");
      el.contentFileMode.classList.remove("hidden");
      el.contentLocalMode.classList.add("hidden");
    });

    el.tabLocalMode.addEventListener("click", () => {
      el.tabLocalMode.classList.add("active");
      el.tabFileMode.classList.remove("active");
      el.contentLocalMode.classList.remove("hidden");
      el.contentFileMode.classList.add("hidden");
    });

    // File input change
    el.videoFileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        el.fileLabel.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
      }
    });

    // Drag and drop zone
    el.dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      el.dropZone.classList.add("dragover");
    });
    el.dropZone.addEventListener("dragleave", () => {
      el.dropZone.classList.remove("dragover");
    });
    el.dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      el.dropZone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        el.videoFileInput.files = e.dataTransfer.files;
        const file = e.dataTransfer.files[0];
        el.fileLabel.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
      }
    });

    // Ingest submit
    el.ingestForm.addEventListener("submit", handleIngestSubmit);

    // Video Player timeupdate
    el.mainVideoPlayer.addEventListener("timeupdate", () => {
      const cur = el.mainVideoPlayer.currentTime;
      const dur = el.mainVideoPlayer.duration || (state.activeVideoData ? state.activeVideoData.duration_sec : 0);
      el.playerTimeDisplay.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;

      if (dur > 0) {
        const pct = (cur / dur) * 100;
        const playhead = document.getElementById("timeline-playhead");
        if (playhead) {
          playhead.style.left = `${pct}%`;
        }
      }
    });

    // Timeline Click Scrubber
    el.timelineTrackContainer.addEventListener("click", (e) => {
      if (!state.activeVideoData) return;
      const rect = el.timelineTrackContainer.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, clickX / rect.width));
      const targetTime = ratio * (state.activeVideoData.duration_sec || el.mainVideoPlayer.duration || 0);

      seekVideo(targetTime);

      // Check if clicked inside an episode
      const episodes = state.activeVideoData.episodes || [];
      const matchingEp = episodes.find((ep) => targetTime >= ep.start_time_sec && targetTime <= ep.end_time_sec);
      if (matchingEp) {
        selectEpisode(matchingEp.episode_index);
      }
    });

    // Storyboard Zoom Button & Click
    el.btnZoomStoryboard.addEventListener("click", () => {
      if (el.inspectorStoryboardImg.src) {
        openModal(el.inspectorStoryboardImg.src, el.inspectorEpTitle.textContent);
      }
    });
    el.inspectorStoryboardImg.addEventListener("click", () => {
      if (el.inspectorStoryboardImg.src) {
        openModal(el.inspectorStoryboardImg.src, el.inspectorEpTitle.textContent);
      }
    });

    // Modal close
    el.btnCloseModal.addEventListener("click", closeModal);
    el.modalBackdrop.addEventListener("click", closeModal);
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeModal();
    });

    // RAG Filters Toggle
    el.btnToggleFilters.addEventListener("click", () => {
      el.ragFiltersPanel.classList.toggle("hidden");
    });

    // RAG Search & Ask buttons
    el.btnRunSearch.addEventListener("click", handleSearch);
    el.btnRunAsk.addEventListener("click", handleAsk);

    // Enter key in textarea triggers ask (with Ctrl+Enter or Cmd+Enter)
    el.ragQueryInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleAsk();
      }
    });
  }

  /* ==========================================================================
     Application Initialization
     ========================================================================== */

  async function init() {
    setupEventListeners();
    await fetchStats();
    await fetchVideos();

    // If videos exist, load the first one by default
    if (state.videoList.length > 0) {
      await loadVideo(state.videoList[0].video_id);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();

