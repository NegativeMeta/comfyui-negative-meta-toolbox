import { app } from "../../scripts/app.js";

function upstreamDirector(output, link, visited = new Set()) {
    if (!Array.isArray(link) || visited.has(String(link[0]))) return null;
    const id = String(link[0]);
    visited.add(id);
    const node = output[id];
    if (!node) return null;
    if (node.class_type === "MiniMaxH3Director") return node;
    for (const value of Object.values(node.inputs || {})) {
        const found = upstreamDirector(output, value, visited);
        if (found) return found;
    }
    return null;
}

app.registerExtension({
    name: "NegativeMeta.H3ContinuationCanvas",
    setup() {
        const original = app.graphToPrompt;
        app.graphToPrompt = async function (...args) {
            const result = await original.apply(this, args);
            for (const node of Object.values(result.output || {})) {
                if (node.class_type !== "NegativeMeta_H3ContinuationCanvas") continue;
                const director = upstreamDirector(result.output, node.inputs.guide);
                if (!director) throw new Error("H3 clip aspect: cannot find the connected MiniMax H3 Director.");
                const timeline = JSON.parse(director.inputs.timeline_data || "{}");
                node.inputs.director_settings = JSON.stringify({
                    resolution: timeline.resolution || {},
                    external_canvas: director.inputs.external_width_overwrite != null || director.inputs.external_height_overwrite != null,
                });
            }
            return result;
        };
    },
});
