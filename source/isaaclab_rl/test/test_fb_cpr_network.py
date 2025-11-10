import os
os.environ["ENABLE_ISAACLAB"] = "False"
from isaaclab_rl.rsl_rl.fb_cpr.fb_networks import test_fb_networks

if __name__ == "__main__":
    test_fb_networks()