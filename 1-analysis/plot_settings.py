# %% Plot settings for Sample Prep Paper
import matplotlib as mpl

# %%
# mpl.rcdefaults() # to restor to default settings
def samplePrepPaper():
    mpl.rcParams['font.size'] = 8 #12
    #mpl.rcParams['legend', fontsize=2] # does it work at all?
    mpl.rcParams['font.family'] = 'Trebuchet MS'
    mpl.rcParams['figure.dpi'] = 300
    # mpl.rcParams['savefig.transparent'] = True # transparent background
    # mpl.rcParams["figure.figsize"] = (16, 4)
    
    #mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color= ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']) 

    mpl.rcParams['figure.subplot.left'] = .2
    mpl.rcParams['figure.subplot.right'] = .8
    mpl.rcParams['figure.subplot.bottom'] = .2
    mpl.rcParams['figure.subplot.top'] = .8
    # mpl.rcParams['axes.linewidth'] = 0.1 # not working

    #mpl.rcParams["axes.prop_cycle"] #(default: cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])).

# Colour palette from CB
#colorP = [[230, 25, 75, 255], [60, 180, 75, 255], [255, 255, 25, 255], [0, 130, 200, 255], [245, 130, 48, 255], \
          #[145, 30, 180, 255], [7, 240, 240, 255], [240, 50, 230, 255], [210, 245, 60, 255], [250, 190, 190, 255], \
          #[0, 128, 128, 255], [230, 190, 255, 255], [170, 110, 40, 255], [255, 250, 200, 255], [128, 0, 0, 255], [170, 255, 195, 255]]
#colorP = np.array(colorP)/255


# Utilities:
# 1) hide axis spine
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# 2) avoid label cropping when saving fig
# plt.savefig('figs/v3/sampleVol_by_group.png', bbox_inches = 'tight')

# 3) squar axis
# ax.set_box_aspect(1)

# 3) change frame(spine) line width
# frame_lineWidth = 1
#for axis in ['top','bottom','left','right']:
    #cax.spines[axis].set_linewidth(frame_lineWidth)
# change legend frame
#leg = ax.legend(arteList, fontsize = font_legend, frameon = True)
#leg.get_frame().set_linewidth(lineWidth_legend)
