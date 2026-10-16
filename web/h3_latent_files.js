import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

function resize(node) {
    requestAnimationFrame(() => {
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], size[0]), Math.max(node.size[1], size[1])]);
        app.graph.setDirtyCanvas(true, false);
    });
}

function button(node, label, callback) {
    const widget = node.addWidget("button", label, null, callback, { serialize: false });
    widget.serialize = false;
    widget.options ??= {}; widget.options.serialize = false;
    return widget;
}

app.registerExtension({
    name: "NegativeMeta.H3LatentFiles",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "NegativeMeta_H3LatentFileSave") {
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                created?.apply(this, arguments);
                button(this, "💾 Download Latent", () => {
                    const file = this._negativeMetaLatentFile;
                    if (!file) { alert("Generate a clip first, then download its latent."); return; }
                    const params = new URLSearchParams({filename: file.filename,
                        subfolder: file.subfolder, type: "temp", t: String(Date.now())});
                    const link = document.createElement("a");
                    link.href = api.apiURL(`/view?${params}`);
                    link.download = file.download_name || file.filename;
                    document.body.appendChild(link); link.click(); link.remove();
                });
                resize(this);
            };
            const executed = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                executed?.apply(this, arguments);
                const file = message?.h3_latent_file?.[0];
                if (file && file.type === "temp" && typeof file.filename === "string") {
                    this._negativeMetaLatentFile = file;
                }
                const path = message?.text?.[0];
                if (typeof path !== "string" || !path) return;
                let widget = this.widgets?.find(w => w.name === "temporary_latent_path");
                if (!widget) {
                    widget = ComfyWidgets.STRING(this, "temporary_latent_path",
                        ["STRING", {multiline: true}], app).widget;
                    widget.serialize = false;
                    widget.options ??= {}; widget.options.serialize = false;
                    if (widget.inputEl) widget.inputEl.readOnly = true;
                }
                widget.value = path; resize(this);
            };
        }
        if (nodeData.name === "NegativeMeta_H3LatentFileLoad") {
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                created?.apply(this, arguments);
                const node = this;
                button(node, "📂 Select Latent File", () => {
                    const input = document.createElement("input");
                    input.type = "file"; input.accept = ".safetensors";
                    input.style.display = "none"; input.oncancel = () => input.remove();
                    input.onchange = async () => {
                        const file = input.files?.[0];
                        if (!file) { input.remove(); return; }
                        try {
                            if (!file.name.toLowerCase().endsWith(".safetensors")) throw new Error("Choose a NegativeMeta H3 AV .safetensors file.");
                            const body = new FormData();
                            body.append("image", file); body.append("type", "input");
                            body.append("subfolder", "negative_meta_h3_latents");
                            const response = await api.fetchApi("/upload/image", {method: "POST", body});
                            if (!response.ok) throw new Error(`Upload failed (${response.status}). ${response.status === 413 ? "The server upload limit was exceeded; use the absolute File Path instead." : "The previous selection was kept."}`);
                            const uploaded = await response.json();
                            if (uploaded.type !== "input" || !uploaded.name) throw new Error("Invalid upload response.");
                            const widget = node.widgets.find(w => w.name === "file_path");
                            if (!widget) throw new Error("File Path widget is unavailable.");
                            widget.value = `${uploaded.subfolder ? uploaded.subfolder + "/" : ""}${uploaded.name} [input]`;
                            widget.callback?.(widget.value); resize(node);
                        } catch (error) { alert(String(error.message || error)); }
                        finally { input.remove(); }
                    };
                    document.body.appendChild(input); input.click();
                });
                button(node, "↩ Use Last Temporary Latent", () => {
                    const widget = node.widgets.find(w => w.name === "file_path");
                    if (widget) { widget.value = ""; widget.callback?.(""); }
                    app.graph.setDirtyCanvas(true, false);
                });
                resize(node);
            };
        }
    },
});
