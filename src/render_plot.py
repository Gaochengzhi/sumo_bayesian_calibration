from __future__ import annotations
import os, glob, cv2, imageio
from IPython.display import Image, display  # 如不用 Jupyter 可删


# ------------------------------------------------------------
def video_to_gif(
    video_path: str,
    out_dir: str = "./out",
    n_frames: int = 10,  # 要抽取的帧数
    step_sec: float = 0.5,  # 连续帧间隔(秒)
    start_sec: float = 0,  # 从视频第几秒开始
    crop_ratios=(0, 0, 0, 0),  # 裁剪(上、下、左、右)百分比
    out_h: int = 380,  # GIF 高度，宽度按比例自适应
    jpg_quality: int = 80,  # 保存 jpg 质量；不想保 jpg 可设为 0
    prefix: str = "frame_",  # 文件名前缀
    gif_fps: float = 15,  # GIF 播放帧率(帧/秒)
):
    """抽帧并生成 GIF"""
    os.makedirs(out_dir, exist_ok=True)
    _clean(out_dir, prefix)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("无法打开视频")

    fps = cap.get(cv2.CAP_PROP_FPS)
    step_frame = max(1, int(fps * step_sec))
    start_frame = int(fps * start_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for idx in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + idx * step_frame)
        ok, frame = cap.read()
        if not ok:
            break
        frame = _process(frame, crop_ratios, out_h)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转 RGB 给 imageio
        if jpg_quality:  # 如不保存 jpg 可提前把 jpg_quality 设 0
            cv2.imwrite(
                f"{out_dir}/{prefix}{idx:02d}.jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpg_quality],
            )
    cap.release()

    if frames:
        _save_gif(frames, f"{out_dir}/{prefix}animation.gif", gif_fps)


# ------------------------------------------------------------
def _process(img, ratios, out_h):
    """裁剪 + 等比缩放"""
    t, b, l, r = ratios
    h, w = img.shape[:2]
    img = img[int(h * t) : h - int(h * b), int(w * l) : w - int(w * r)]
    out_w = int(out_h * (img.shape[1] / img.shape[0]))
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _save_gif(frames, path, fps):
    with imageio.get_writer(path, mode="I", duration=1 / fps, loop=0) as wr:
        for f in frames:
            wr.append_data(f)
    try:
        display(Image(path))
    except:
        print("GIF saved ->", path)


def _clean(out_dir, prefix):
    for p in glob.glob(f"{out_dir}/{prefix}*.*"):
        os.remove(p)


import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import imageio.v2 as imageio
import traci

# Notebook 实时显示
try:
    from IPython.display import display, clear_output

    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False


def _in_notebook():
    if not _HAS_IPYTHON:
        return False
    try:
        from IPython import get_ipython

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


# ---------- utils ----------
def orientation_tri(
    center: np.ndarray, yaw_deg: float, side: float = 1.2
) -> np.ndarray:
    a = np.deg2rad(90 - yaw_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = np.array([[side, 0.0], [-0.5 * side, 0.5 * side], [-0.5 * side, -0.5 * side]])
    return pts @ rot.T + center


def vehicle_box(center, yaw, length, width):
    a = np.deg2rad(90 - yaw)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    l, w = length / 2, width / 2
    pts = np.array([[-l, -w], [l, -w], [l, w], [-l, w]])
    return pts @ rot.T + center


# ---------- renderer ----------
class SumoMatplotlibRenderer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._init_defaults()
        self._prep_dirs()

        self.fig, self.ax = plt.subplots(figsize=cfg["fig_size"], dpi=400)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_facecolor(self.colors["bg"])

        self._draw_road_map()
        self.hist = {}  # 车辆历史
        self.frame_id = 0
        self._dyn_artists = []  # 动态元素容器
        self.live = _in_notebook()

    # ------- 默认参数 -------
    def _init_defaults(self):
        self.colors = dict(
            bg=self.cfg.get("bg_color", "white"),
            bus=self.cfg.get("bus_color", "#006CFF"),
            car=self.cfg.get("car_color", "#9FA4AD"),
            trail=self.cfg.get("convoy_color", "#3AA0FF"),
            lane_fill=self.cfg.get("lane_fill", "white"),
            lane_edge=self.cfg.get("lane_edge_dark", "black"),
            lane_mid=self.cfg.get("lane_mid_light", "#C0C4CC"),
        )
        self.lane_lw = self.cfg.get("lane_lw", 0.2)
        self.hist_len = self.cfg.get("hist_len", 20)
        self.hist_every = self.cfg.get("hist_every", 3)
        self.veh_alpha = self.cfg.get("veh_alpha", 0.3)

    # ------- 静态底图 -------
    def _draw_road_map(self):
        fill, edge, mid = (
            self.colors["lane_fill"],
            self.colors["lane_edge"],
            self.colors["lane_mid"],
        )
        lw = self.lane_lw

        # 记录已绘制的边界线坐标
        drawn_lines = set()

        for lid in traci.lane.getIDList():
            shape = np.array(traci.lane.getShape(lid))
            w = traci.lane.getWidth(lid) or 3.2
            left = self._offset(shape, +w / 2)
            right = self._offset(shape, -w / 2)
            poly = np.vstack((left, right[::-1]))

            self.ax.add_patch(
                Polygon(poly, facecolor=fill, edgecolor=None, alpha=0.75, zorder=0)
            )

            # 总是绘制左右边界线，确保连接部分也能显示
            self.ax.plot(left[:, 0], left[:, 1], color=edge, lw=lw, zorder=1)
            self.ax.plot(right[:, 0], right[:, 1], color=edge, lw=lw, zorder=1)

        vx, vy = self.cfg.get("view_x", 120) / 2, self.cfg.get("view_y", 80) / 2

        # 考虑裁剪比例
        crop_left = self.cfg.get("crop_left_ratio", 0)
        crop_right = self.cfg.get("crop_right_ratio", 0)
        crop_bottom = self.cfg.get("crop_bottom_ratio", 0)
        crop_top = self.cfg.get("crop_top_ratio", 0)

        # 调整视图范围
        x_min = -vx + vx * crop_left
        x_max = vx - vx * crop_right
        y_min = -vy + vy * crop_bottom
        y_max = vy - vy * crop_top

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

    # ------- 抓取车辆 -------
    def collect_states(self):
        return {
            vid: dict(
                pos=np.array(traci.vehicle.getPosition(vid)),
                yaw=traci.vehicle.getAngle(vid),
                length=traci.vehicle.getLength(vid),
                width=traci.vehicle.getWidth(vid),
            )
            for vid in traci.vehicle.getIDList()
        }

    # ------- 渲染单帧 -------
    def render_frame(self, states, sim_t: float):
        # 清除上帧动态 artist

        for art in self._dyn_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._dyn_artists.clear()

        # 更新历史
        if self.frame_id % self.hist_every == 0:
            for vid, st in states.items():
                self.hist.setdefault(vid, []).append(st.copy())
                self.hist[vid] = self.hist[vid][-self.hist_len :]

        # 拖影
        for vid, h in self.hist.items():
            n = len(h) - 1
            if n <= 0:
                continue
            for idx, st in enumerate(h[:-1][::-1]):  # 旧 -> 新
                # alpha 从 0 到 self.veh_alpha，idx=0时为0，idx=n-1时接近self.veh_alpha
                a = self.veh_alpha - self.veh_alpha * (idx / n)
                color = self.colors["bus"] if st["length"] > 6.5 else self.colors["car"]
                box = vehicle_box(st["pos"], st["yaw"], st["length"], 2)
                poly = Polygon(box, facecolor=color, edgecolor=None, alpha=a, zorder=2)
                self.ax.add_patch(poly)
                self._dyn_artists.append(poly)

        # 当前帧车辆
        for st in states.values():
            color = self.colors["bus"] if st["length"] > 6.5 else self.colors["car"]
            box = vehicle_box(st["pos"], st["yaw"], st["length"], 2)
            poly = Polygon(
                box,
                facecolor=color,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.4,
                zorder=3,
            )
            tri = Polygon(
                orientation_tri(st["pos"], st["yaw"]),
                facecolor="black",
                edgecolor=None,
                zorder=4,
            )
            self.ax.add_patch(poly)
            self.ax.add_patch(tri)
            self._dyn_artists.extend([poly, tri])

        # 保存
        base = f"{self.cfg['prefix']}{sim_t:.1f}"
        self.fig.savefig(
            os.path.join(self.pdf_dir, f"{base}.pdf"), bbox_inches="tight", pad_inches=0
        )
        self.fig.savefig(
            os.path.join(self.png_dir, f"{base}.png"),
            dpi=400,
            bbox_inches="tight",
            pad_inches=0,
        )

        # notebook 实时显示
        if self.live:
            clear_output(wait=True)
            display(self.fig)

        self.frame_id += 1

    # ------- 合成 GIF -------
    def genrate_gif(self):
        pngs = sorted(
            glob.glob(os.path.join(self.png_dir, f"{self.cfg['prefix']}*.png"))
        )
        if not pngs:
            return
        imgs = [imageio.imread(p) for p in pngs]
        gif_path = os.path.join(self.cfg["output_dir"], self.cfg["gif_filename"])
        imageio.mimsave(
            gif_path, imgs, duration=self.cfg["gif_frame_duration"], loop=0
        )  # loop=0 表示无限循环
        if self.cfg.get("cleanup_frames_after_gif", True):
            for p in pngs:
                os.remove(p)

    # ------- 工具 -------
    def _offset(self, poly, dist):
        res = []
        for i in range(len(poly) - 1):
            p0, p1 = poly[i], poly[i + 1]
            seg = p1 - p0
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-6:
                continue
            n = np.array([-seg[1], seg[0]]) / seg_len
            res.append(p0 + dist * n)
            if i == len(poly) - 2:
                res.append(p1 + dist * n)
        return np.array(res)

    def _prep_dirs(self):
        self.pdf_dir = os.path.join(
            self.cfg["output_dir"], self.cfg.get("pdf_output_dir", "pdf")
        )
        self.png_dir = os.path.join(self.cfg["output_dir"], "frames/tmp_png")
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.png_dir, exist_ok=True)
