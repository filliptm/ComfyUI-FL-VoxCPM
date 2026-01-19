import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FL-VoxCPM.appearance",
    async nodeCreated(node) {
        // Check if the node's comfyClass starts with "FL_VoxCPM"
        if (node.comfyClass.startsWith("FL_VoxCPM")) {
            // Apply styling - same colors as Fill-Nodes for consistency
            node.color = "#16727c";
            node.bgcolor = "#4F0074";
        }
    }
});
