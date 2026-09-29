// Inlined before main.js on the destination chat page. Keep the handoff in sync
// with react/src/features/chat/hooks/chatSocketBootstrap.ts.
(function () {
    var convoId = document.currentScript.dataset.convoId;
    if (!convoId || window.aquiChatSocketBootstrap) return;
    var socket;
    var startedAt = Date.now();
    try {
        var protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
        socket = new WebSocket(protocol + window.location.host + '/ws/convo/' + encodeURIComponent(convoId) + '/');
    } catch (_) {
        return; // The hook owns the normal retry policy, including constructor failures.
    }

    var events = [];
    var characters = 0;
    var claimed = false;
    var expired = false;
    var permanentClose = null;
    var fatalReason = '';
    var expiryTimer = setTimeout(expire, 15000);

    function detach() {
        socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
        clearTimeout(expiryTimer);
        window.removeEventListener('pagehide', dispose);
    }

    function keepTerminalFailure() {
        events = [];
        characters = 0;
        if (fatalReason) events.push({
            type: 'message',
            event: new MessageEvent('message', { data: JSON.stringify({ exception: fatalReason, fatal: true }) })
        });
        if (permanentClose) events.push({ type: 'close', event: permanentClose });
    }

    function expire() {
        expired = true;
        clearTimeout(expiryTimer);
        // Never replay a truncated snapshot/stream. Keep only a small terminal
        // tombstone so a slow bundle cannot turn a permanent failure into a retry.
        keepTerminalFailure();
        socket.onopen = socket.onmessage = socket.onerror = null;
        if (socket.readyState === WebSocket.CLOSED) detach();
        else socket.close();
    }

    function dispose() {
        if (claimed) return;
        claimed = true;
        detach();
        events = [];
        if (socket.readyState !== WebSocket.CLOSED) socket.close();
        if (window.aquiChatSocketBootstrap === handoff) delete window.aquiChatSocketBootstrap;
    }

    function buffer(type, event) {
        if (claimed || expired) return;
        var size = type === 'message' && typeof event.data === 'string' ? event.data.length : 0;
        // Up to 8 MiB of UTF-16 text, including sizeable saved conversations.
        if (events.length >= 512 || characters + size > 4 * 1024 * 1024) {
            expire();
            return;
        }
        characters += size;
        events.push({ type: type, event: event });
    }

    socket.onopen = function (event) { buffer('open', event); };
    socket.onerror = function (event) { buffer('error', event); };
    socket.onmessage = function (event) {
        // The hook parses conversation data. Only inspect possible fatal payloads
        // here, retaining the reason even if a large debug page accompanies it.
        if (typeof event.data === 'string' && event.data.indexOf('"fatal"') !== -1) {
            try {
                var data = JSON.parse(event.data);
                if (data.fatal && typeof data.exception === 'string') fatalReason = data.exception.slice(0, 4096);
            } catch (_) { /* The hook reports malformed messages after adoption. */ }
        }
        buffer('message', event);
    };
    socket.onclose = function (event) {
        if (event.code === 4401 || event.code === 4404 || event.code === 4409) permanentClose = event;
        if (expired) {
            keepTerminalFailure();
            detach();
        } else buffer('close', event);
    };

    var handoff = {
        take: function (requestedConvoId) {
            if (claimed) return null;
            if (requestedConvoId !== convoId || (expired && !permanentClose && socket.readyState === WebSocket.CLOSED)) {
                dispose();
                return null;
            }
            claimed = true;
            detach();
            if (window.aquiChatSocketBootstrap === handoff) delete window.aquiChatSocketBootstrap;
            var buffered = events;
            events = [];
            return { socket: socket, events: buffered, startedAt: startedAt };
        }
    };
    window.aquiChatSocketBootstrap = handoff;
    window.addEventListener('pagehide', dispose);
})();
