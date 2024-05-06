import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors


def common_draw(Ys: list, labels: list, y_tick_tuple: tuple, x_lim_tuple: tuple, x_name: list = None,
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
                         linestyle=linestyles[i % 4],
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
        # ax.legend(prop={'size': legendsize}, loc=legend_pos,
        #           ncol=lengent_ncol, columnspacing=columnspacing)

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


# draw_vary_ray_num()
time_distribution()