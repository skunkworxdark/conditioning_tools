# `conditioning_tools` for InvokeAI (v4.0+)

 A collection of experimental custom nodes for manipulating conditioning tensors

Discord Link: [`conditioning_tools`](https://discord.com/channels/1020123559063990373/1352361385060860025) for support, discussion, and feedback.

## Nodes

- `Flux Redux Downsampling` - Downsampling or weighting a flux redux conditioning -  this is derived from 


## Usage

#### <ins>Install</ins><BR>
There are two options to install the nodes:

1. **Recommended**: Git clone into the `invokeai/nodes` directory. This allows updating via `git pull`.

    - Open your terminal or command prompt and navigate to your InvokeAI "nodes" folder.
    - Run the following command::
    ```bash
    git clone https://github.com/skunkworxdark/conditioning_tools.git
    ```

2. Manually download [conditioning_tools.py](conditioning_tools.py) & [__init__.py](__init__.py) then place them in a subfolder (e.g., `conditioning_tools`) under `invokeai/nodes`. 

#### <ins>Update</ins><BR>
Run a `git pull` from the `conditioning_tools` folder.

Or run the provided `update.bat`(windows) or `update.sh`(Linux).

For manual installs, download and replace the files.

#### <ins>Remove</ins><BR>
Delete the `conditioning_tools` folder. Or rename it to `_conditioning_tools`` so InvokeAI will ignore it.

## ToDo
- better readme
- Add more useful latent tools ....

# Example Usage
ToDo