import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors


def common_draw(Ys: list, labels: list, y_tick_tuple: tuple, x_lim_tuple: tuple, x_name: list = None,
                saveName: str = "pic/paper/temp.pdf",
                colors=[],
                colorOffset=0,
                show_legend=True,
                legendsize=20,
                legend_pos=2,
                x_axis_name="",
                y_axis_name='Time (ms)',
                y_log=False,
                ymin=-1,
                ymax=-1,
                lengent_ncol=3,
                selfdef_figsize=(12, 6),
                BAR=True,
                bar_width=0.8,
                rBAR=False,
                uptext=False,
                columnspacing=1.6,
                rYs=[],
                ry_axis_name='',
                r_labels=[],
                r_ymax=-1,
                common_font_size=26,
                x_num=0,
                line=0,
                y_ticks=[],
                y_label_fontsize=''):
    with PdfPages(saveName) as pdf:
        font = {
            "weight": "normal",
            "size": common_font_size,
        }

        if x_num == 0:
            x_num = len(x_name)

        X = np.arange(x_num) + 1
        if colors == []:
            colors = ["#e64b35", "gold", "dodgerblue",
                      "lightgreen", "cyan", "green", "chocolate"]

        markers = ["o", "^", "D", "s", "*", "P", "x"]
        linestyles = ["-", ":", "-.", "--"]
        markerfacecolors = ["black", "none"]

        plt.figure(figsize=selfdef_figsize)
        if not BAR:
            for i in range(0, len(Ys)):
                plt.plot(X,
                         Ys[i],
                         label=labels[i],
                         linestyle=linestyles[i],
                         color=colors[i],
                         marker=markers[i],
                         # markerfacecolor=markerfacecolors[i%2],
                         markersize=10,
                         linewidth=3,
                         )
        else:
            hatches = ['//','--||','\\\\','--','o', 'x',  '+', '*', 'O', ]
            X = np.arange(x_num) + 1
            total_width, n = bar_width, len(Ys)
            width = total_width / n
            X = X - (total_width - width) / 2
            for i in range(0, len(Ys)):
                plt.bar(X + width * i, Ys[i], width=width, label=labels[i],
                        color=colors[i % len(colors) + colorOffset],
                        edgecolor="black")

        xlim_left, xlim_right = x_lim_tuple
        if line != 0:
            line_x = np.linspace(xlim_left, xlim_right, 4)
            line_y = [line, line, line, line]
            plt.plot(line_x, line_y, color="darkgreen",
                     label="RTc3 Uniform",  linestyle='--')
        if y_log:
            plt.yscale("log")
        x_ticks = np.linspace(1, x_num, len(x_name))
        if x_name == None:
            x_name = [str(float(x)/10) for x in range(0, x_num)]
        plt.xticks(x_ticks, x_name, fontsize=common_font_size)

        # specify y_ticks
        # y_ticks = np.linspace(0,500,20)
        # y_len, y_num = y_tick_tuple
        # y_ticks = np.linspace(0, y_len, y_num)
        # plt.yticks(y_ticks, fontsize=common_font_size)

        # adaptively
        plt.yticks(fontsize=common_font_size)
        if y_ticks != []:
            plt.yticks(y_ticks[0], y_ticks[1])
        if ymax != -1:
            plt.ylim(ymax=ymax)
        if ymin != -1:
            plt.ylim(ymin=ymin)

        if uptext:
            # * batch range query
            for j in range(2, len(X)):
                plt.text(X[j] + width * 1, Ys[1][j] + 10, 'N/A',
                         ha='center', va='bottom', rotation=90)

        ax = plt.gca()

        if x_axis_name != "":
            ax.set_xlabel(x_axis_name, font)
        if y_label_fontsize == '':
            ax.set_ylabel(y_axis_name, font)
        else:
            ax.set_ylabel(
                y_axis_name, {"weight": "normal", "size": y_label_fontsize})
        if show_legend:
            ax.legend(prop={'size': legendsize}, loc=legend_pos,
                    ncol=lengent_ncol, columnspacing=columnspacing)

        if rYs != []:
            ax2 = plt.twinx()
            for i in range(0, len(rYs)):
                ax2.plot(X[1:],
                         rYs[i],
                         label=r_labels[i],
                         linestyle=linestyles[(i+2) % 4],
                         color=colors[i+2],
                         marker=markers[i+2],
                         # markerfacecolor=markerfacecolors[i%2],
                         markersize=10,
                         linewidth=3,
                         )
            if y_log:
                ax2.set_yscale("log")
            if r_ymax != -1:
                ax2.set_ylim(ymax=r_ymax)
            ax2.tick_params(axis='y', labelsize=common_font_size)
            ax2.set_ylabel(ry_axis_name, font)
            ax2.legend(prop={'size': legendsize}, loc='upper right',
                       ncol=lengent_ncol, columnspacing=columnspacing)
        plt.tight_layout()
        pdf.savefig()

def draw_memory(Ys: list, labels: list, y_tick_tuple: tuple, x_lim_tuple: tuple, x_name: list = None,
                saveName: str = "pic/paper/temp.pdf",
                colors=[],
                colorOffset=0,
                legendsize=20,
                legend_pos=2,
                x_axis_name="",
                y_axis_name='Time (ms)',
                y_log=False,
                ymin=-1,
                ymax=-1,
                lengent_ncol=3,
                selfdef_figsize=(12, 6),
                BAR=True,
                bar_width=0.8,
                rBAR=False,
                uptext=False,
                columnspacing=1.6,
                rYs=[],
                ry_axis_name='',
                r_labels=[],
                r_ymax=-1,
                common_font_size=26,
                x_num=0,
                line=0,
                y_ticks=[],
                y_label_fontsize=''):
    with PdfPages(saveName) as pdf:
        font = {
            "weight": "normal",
            "size": common_font_size,
        }
        if x_num == 0:
            x_num = len(x_name)
        X = np.arange(x_num) + 1
        plt.figure(figsize=selfdef_figsize)
        total_width, n = bar_width, len(Ys)
        width = total_width / n
        X = X - (total_width - width) / 2
        handles = []
        for i in range(0, n - 1):
            h = plt.bar(X + width * i, Ys[i], width=width, label=labels[i],
                    color=colors[i % len(colors) + colorOffset],
                    edgecolor="black")
            handles.append(h)
        h = plt.bar(X + width * (n - 2), Ys[n - 1], bottom=Ys[n - 2], width=width, label=labels[n - 1],
                color=colors[n - 1],
                edgecolor="black")
        handles.append(h)
        if y_log:
            plt.yscale("log")
        x_ticks = np.linspace(1, x_num, len(x_name))
        if x_name == None:
            x_name = [str(float(x)/10) for x in range(0, x_num)]
        plt.xticks(x_ticks, x_name, fontsize=common_font_size)

        # adaptively
        plt.yticks(fontsize=common_font_size)
        if y_ticks != []:
            plt.yticks(y_ticks[0], y_ticks[1])
        if ymax != -1:
            plt.ylim(ymax=ymax)
        if ymin != -1:
            plt.ylim(ymin=ymin)
        ax = plt.gca()
        if x_axis_name != "":
            ax.set_xlabel(x_axis_name, font)
        if y_label_fontsize == '':
            ax.set_ylabel(y_axis_name, font)
        else:
            ax.set_ylabel(
                y_axis_name, {"weight": "normal", "size": y_label_fontsize})
        
        new_handles = [handles[0], handles[3], handles[1], handles[4], handles[2]]
        ax.legend(handles=new_handles, prop={'size': legendsize}, loc=legend_pos,
                  ncol=lengent_ncol, columnspacing=columnspacing)

        plt.tight_layout()
        pdf.savefig()

def draw_stacking(Ys: list, labels: list, x_lim_tuple: tuple, x_name: list = None,
                  saveName: str = "pic/paper/temp.pdf",
                  colors=[],
                  colorOffset=0,
                  legendsize=26,
                  legend_pos=2,
                  x_axis_name="Selectivity",
                  y_axis_name='Time (ms)',
                  y_log=False,
                  ymax=-1,
                  lengent_ncol=3,
                  selfdef_figsize=(12, 6),
                  BAR=True,
                  uptext=False,
                  columnspacing=1.6,
                  common_font_size=26):
    with PdfPages(saveName) as pdf:
        font = {
            "weight": "normal",   # "bold"
            "size": common_font_size,
        }
        x_num = len(x_name)

        print(f"[INFO] use x_num: {x_num}")

        X = np.arange(x_num) + 1
        if colors == []:
            # colors = ["#e64b35","gold","dodgerblue","lightgreen","cyan","green","chocolate"]
            colors = ["lightgreen", 'xkcd:jade',
                      'dodgerblue', 'gold', '#e64b35']

        markers = ["o", "^", "D", "s", "*", "P", "x"]
        linestyles = ["-", ":", "-.", "--"]
        markerfacecolors = ["black", "none"]

        plt.figure(figsize=selfdef_figsize)
        hatches = ['\\', '.', '/', '-', 'o', 'x',  '+', '*', 'O', ]
        X = np.arange(x_num) + 1
        width = 0.65
        print(len(Ys), len(X))
        print(labels)

        accumulated_height = []
        accumulated_height.append(Ys[0])
        for i in range(1, len(Ys)):
            cur_height = []
            for j in range(x_num):
                cur_height.append(Ys[i][j] + accumulated_height[i - 1][j])
            accumulated_height.append(cur_height)
        accumulated_height.insert(0, [0] * x_num)

        # set bottom
        for i in range(0, len(Ys)):
            plt.bar(X, Ys[i], width=width, label=labels[i],
                    bottom=accumulated_height[i],
                    color=colors[i % len(colors) + colorOffset],
                    edgecolor="black")

        xlim_left, xlim_right = x_lim_tuple
        if y_log:
            plt.yscale("log")
        print(plt.xlim(xlim_left, xlim_right))
        print(plt.ylim())
        x_ticks = np.linspace(1, x_num, x_num)
        print(x_ticks)
        if x_name == None:
            x_name = [str(float(x)/10) for x in range(0, x_num)]
        plt.xticks(x_ticks, x_name, fontsize=common_font_size)

        # adaptively
        plt.yticks(fontsize=common_font_size)
        if ymax != -1:
            plt.ylim(ymax=ymax)

        if uptext:
            # * batch range query
            for j in range(2, len(X)):
                plt.text(X[j] + width * 1, Ys[1][j] + 10, 'N/A',
                         ha='center', va='bottom', rotation=90)

        ax = plt.gca()

        if x_axis_name != "":
            ax.set_xlabel(x_axis_name, font)
        ax.set_ylabel(y_axis_name, font)
        ax.legend(prop={'size': legendsize}, loc=legend_pos,
                  ncol=lengent_ncol, columnspacing=columnspacing)
        plt.tight_layout()
        pdf.savefig()


def draw_vary_ray_num():
    Y = [
        [2063293.594, 18871485.19, 102663080.3, 139814773.4, 141472064],
    ]
    Y_label = ["STK"]
    X_tick_name = ["1k", "10k", "100k", "1000k", "3000k"] 
    common_draw(Y, Y_label, None, (0.5, 0.5 + len(X_tick_name)),
                X_tick_name, "pic/vary_ray_num.pdf",
                colors=["dodgerblue"],
                colorOffset=0,
                x_axis_name='Number of rays',
                y_axis_name='RPS',
                y_log=True,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=3,
                # ymax=140,
                BAR=False,
                selfdef_figsize=(8,6))


def time_distribution():
    record_x_name = ['GAU', 'STK', 'TAO']
    record_labels = [
        'Transferring Data',
        'Updating Units',
        'Building the BVH tree',
        'Detecting Outliers',
    ]
    time = [
        [0.025119, 0.023305, 0.0125783],
        [0.103257, 0.085829, 0.0161487],
        [0.190374, 0.189113, 0.168299],
        [0.168199, 0.089309, 0.198239],
    ]
    draw_stacking(time,
                  record_labels,
                  x_lim_tuple=(0.5, 0.5 + len(record_x_name)),
                  x_name=record_x_name,
                  saveName="pic/breakdown-running-time.pdf",
                  colorOffset=0,
                  x_axis_name='Dataset',
                  y_axis_name='Time (ms)',
                  legendsize=26,
                  common_font_size=26,
                  lengent_ncol=2,
                  ymax=0.73,
                  legend_pos=1,
                  #   columnspacing=0.6,
                  #   colors=['#d6a9aa', "lightgreen", 'xkcd:jade', 'dodgerblue', 'gold',],
                  colors=["lightgreen", 'xkcd:jade', 'dodgerblue', 'gold', ],
                  )

def overall_time(colors):
    record_x_name = ["GAU", "STK", "TAO"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [185.267, 88.488, 5.86],
        [5.326, 1.877, 1],
        [165, 9.752, 3.65],
        [0.487, 0.388, 0.395],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/highlight-results-time-86.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Dataset',
                y_axis_name='Time (ms)',
                y_log=True,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                legend_pos='best',
                ymax=240)

def overall_memory(colors):
    record_x_name = ["GAU", "STK", "TAO"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD-Host", "RTOD-Device"]
    mem = [
        [83, 73, 44],
        [39, 36.6, 14.6],
        [42.5, 39.4, 14.6],
        [3.01172, 3.01953, 0.6875],
        [26, 26, 2],
    ]
    draw_memory(mem, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/highlight-results-mem-86.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Dataset',
                y_axis_name='Memory (MB)',
                y_log=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=3,
                legend_pos='best',
                ymax=120)

def vary_window_size(colors):
    # GAU
    record_x_name = ["10k", "50k", "100k", "150k", "200k"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [1948.531347, 640, 186, 183, 179],
        [13.54, 5.98, 6.55, 5.07, 5.22],
        [84.01, 145.61, 163.55, 148.02, 186.75],
        [0.597036, 0.530574, 0.525301, 0.506659, 0.489369],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-window-GAU.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Window Size',
                y_axis_name='Time (ms)',
                y_log=True,
                common_font_size=26,
                columnspacing=0.8,
                show_legend=False,
                legendsize=26,
                lengent_ncol=2,
                legend_pos='upper right',
                ymax=3000,
                BAR=False)

    # STK
    record_x_name = ["10k", "50k", "100k", "150k", "200k"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [49, 60.39, 91.945, 111.071, 134.129],
        [2.016, 1.672, 1.76, 2, 1.87],
        [1.78, 4.39, 11.82, 16.76, 21.85],
        [0.360827, 0.377785, 0.39899, 0.418776, 0.426521],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-window-STK.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Window Size',
                y_axis_name='Time (ms)',
                y_log=True,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                show_legend=False,
                lengent_ncol=4,
                legend_pos='best',
                ymax=200,
                BAR=False)

    # TAO
    record_x_name = ["1k", "5k", "10k", "15k", "20k"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [2.386, 4.187, 5.881, 6.873, 7.9],
        [0.38, 0.644, 0.965, 1.26, 1.616],
        [0.49, 1.74, 3.71, 6.56, 9.35],
        [0.307149, 0.348122, 0.393131, 0.431411, 0.457859],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-window-TAO.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Window Size',
                y_axis_name='Time (ms)',
                y_log=True,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                show_legend=False,
                lengent_ncol=1,
                ymax=12,
                BAR=False)

def vary_slide_size(colors):
    # GAU
    record_x_name = ["5%", "10%", "20%", "50%", "100%"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [196.637, 379.785, 787.896, 2967.92, 17728],
        [5.688, 7.34, 11, 26.3, 49.66],
        [163.08, 94.22, 49.17, 40.18, 65.38],
        [0.5494, 0.65214, 0.848003, 1.48439, 2.60254],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-slide-GAU.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Slide Size/Window Size',
                y_axis_name='Time (ms)',
                y_log=True,
                show_legend=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                ymax=25000)

    # STK
    record_x_name = ["5%", "10%", "20%", "50%", "100%"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [85.358, 166.75, 323.92, 746.948, 1142],
        [1.96, 3.446, 6.4, 16.2, 33.8],
        [9.55, 9.27, 6.36, 12.14, 22.32],
        [0.401412, 0.513108, 0.722111, 1.363, 2.45399],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-slide-STK.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Slide Size/Window Size',
                y_axis_name='Time (ms)',
                y_log=True,
                show_legend=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                ymax=1500)

    # TAO
    record_x_name = ["5%", "10%", "20%", "50%", "100%"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [5.603, 10.867, 20.363, 47.15, 73.246],
        [0.96, 1.265, 1.93, 3.4, 5.7],
        [3.89, 3.33, 3.92, 4.38, 8.59],
        [0.396153, 0.405813, 0.437548, 0.513555, 0.628039],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-slide-TAO.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Slide Size/Window Size',
                y_axis_name='Time (ms)',
                y_log=True,
                show_legend=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                ymax=100)

def vary_R(colors):
    # GAU
    record_x_name = ["25%", "50%", "100%", "500%", "1000%"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [8069, 1690, 188.304, 74.073, 77.69],
        [115.3, 17.7, 5.72, 1.65, 1.46],
        [5859.5, 990.92, 160.03, 8.41, 1.8],
        [1.05287, 0.621829, 0.499456, 0.325994, 0.286911],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-R-GAU.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Distance Threshold',
                y_axis_name='Time (ms)',
                y_log=True,
                show_legend=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                ymax=10000)

    # STK
    record_x_name = ["25%", "50%", "100%", "500%", "1000%"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [268.04, 109.927, 88.085, 169.33, 256.59],
        [4.77, 2.48, 1.96, 1.32, 1.25],
        [95.06, 31.16, 9.51, 1.45, 1.11],
        [0.500329, 0.45761, 0.39688, 0.320794, 0.295767],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-R-STK.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Distance Threshold',
                y_axis_name='Time (ms)',
                y_log=True,
                show_legend=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                ymax=400)

    # TAO
    record_x_name = ["25%", "50%", "100%", "500%", "1000%"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [50.761, 12.229, 5.989, 6.1186, 6.048],
        [21, 4, 0.98, 0.204, 0.129],
        [116.2, 16.76, 3.78, 0.72, 0.21],
        [0.777763, 0.56287, 0.394725, 0.2158, 0.182455],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-R-TAO.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Distance Threshold',
                y_axis_name='Time (ms)',
                y_log=True,
                show_legend=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                ymax=150)

def vary_K(colors):
    # GAU
    record_x_name = ["10", "30", "50", "70", "100"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [117.1270265, 133.5844736, 195.2347218, 592.9089938, 3497.38145],
        [3.25, 4.08, 5.11, 7.05, 9.03],
        [17.66, 56.55, 165.67, 325.47, 538.45],
        [0.301978, 0.407293, 0.547744, 0.614817, 0.79909],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-K-GAU.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Neighbor Threshold',
                y_axis_name='Time (ms)',
                y_log=True,
                show_legend=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                ymax=6000)

    # STK
    record_x_name = ["10", "30", "50", "70", "100"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [78.36877094, 77.14202256, 93.61847055, 123.3528939, 163.0575543],
        [1.467, 1.56, 1.94, 2.23, 2.38],
        [3.22, 8.83, 10.03, 12.48, 20.86],
        [0.288736, 0.35102, 0.399524, 0.450588, 0.543054],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-K-STK.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Neighbor Threshold',
                y_axis_name='Time (ms)',
                y_log=True,
                show_legend=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                ymax=300)

    # TAO
    record_x_name = ["10", "30", "50", "70", "100"]
    record_labels = ["MCOD", "NETS", "MDUAL", "RTOD"]
    total_time = [
        [2.99, 4.214, 5.69, 7.622, 11.269],
        [0.64, 0.855, 0.99, 1.13, 1.4],
        [1.32, 2.48, 3.65, 4.81, 6.65],
        [0.224274, 0.317165, 0.395041, 0.471331, 0.587537],
    ]
    common_draw(total_time, record_labels, None, (0.5, 0.5 + len(record_x_name)),
                record_x_name, "pic/varying-K-TAO.pdf",
                colors=colors,
                colorOffset=0,
                x_axis_name='Neighbor Threshold',
                y_axis_name='Time (ms)',
                y_log=True,
                show_legend=False,
                legendsize=26,
                common_font_size=26,
                columnspacing=0.8,
                lengent_ncol=1,
                ymax=16)

# draw_vary_ray_num()
# time_distribution()
colors = ["#e6b745", "#e64b35", "xkcd:jade", "dodgerblue"] # 9394e7 
# overall_time(colors)
# overall_memory(colors + ["gold"])
# vary_window_size(colors)
# vary_slide_size(colors)
# vary_R(colors)
# vary_K(colors)