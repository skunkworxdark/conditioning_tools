# `conditioning_tools` for InvokeAI (v4.0+)

A collection of experimental custom nodes for manipulating conditioning tensors

Discord Link: [`conditioning_tools`](https://discord.com/channels/1020123559063990373/1352361385060860025) for support, discussion, and feedback.

## Nodes

-   `Flux Redux Downsampling` - Downsampling and/or weighting a flux redux conditioning. This is the same as the core node but with a greater range of scale. This is derived from https://github.com/kaibioinfo/ComfyUI_AdvancedRefluxControl
-   `Scale FLUX Conditioning` - Scales a FLUX conditioning field by a factor.
-   `Scale FLUX Redux Conditioning` - Scales a FLUX Redux conditioning field by a factor.
-   `FLUX Conditioning Math` - Performs a Math operation on two FLUX conditionings.
-   `FLUX Redux Conditioning Math` - Performs a Math operation on two FLUX Redux conditionings.
-   `Normalize FLUX Conditioning` - Normalizes a FLUX conditioning field to a unit vector.
-   `Normalize FLUX Redux Conditioning` - Normalizes a FLUX Redux conditioning field to a unit vector.

## Usage

#### <ins>Install</ins><BR>
There are two options to install the nodes:

1.  **Recommended**: Git clone into the `invokeai/nodes` directory. This allows updating via `git pull`.

    -   Open your terminal or command prompt and navigate to your InvokeAI "nodes" folder.
    -   Run the following command::
        ```bash
        git clone https://github.com/skunkworxdark/conditioning_tools.git
        ```

2.  Manually download [conditioning_tools.py](conditioning_tools.py) & [__init__.py](__init__.py) then place them in a subfolder (e.g., `conditioning_tools`) under `invokeai/nodes`. 

#### <ins>Update</ins><BR>
Run a `git pull` from the `conditioning_tools` folder.

Or run the provided `update.bat`(windows) or `update.sh`(Linux).

For manual installs, download and replace the files.

#### <ins>Remove</ins><BR>
Delete the `conditioning_tools` folder. Or rename it to `_conditioning_tools` so InvokeAI will ignore it.

## ToDo
- better readme.md -  examples usage etc.
- Add more useful conditioning tools ....

# Example Usage
ToDo