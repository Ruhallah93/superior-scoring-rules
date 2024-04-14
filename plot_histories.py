import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

# Healthy_Older_People, Power_consumption
for dataset in ['DriverIdentification', 'ConfLongDemo_JSI', 'Healthy_Older_People', 'Motor_Failure_Time',
                'Power_consumption', 'PRSA2017', 'RSSI', 'User_Identification_From_Walking', 'WISDM']:
    data = pd.read_csv("history/" + str(dataset) + ".csv")

    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    # fig.suptitle(dataset)

    axs[0].plot(data['Step'], data['f1'] * 100, 'tab:orange')
    axs[0].set_ylabel('F1 Score')
    # minor_ticks = np.arange(data['f1'].min() * 100, data['f1'].max() * 100, 10)
    # axs[0].set_yticks(minor_ticks)
    # axs[0].grid(which='major', color='orange', linestyle='--', linewidth=0.5, axis="both", alpha=0.5)
    axs[0].axhline(data['f1'].max() * 100, color='gray', linestyle='dashed', linewidth=0.5)
    axs[0].scatter(data['f1'].argmax(), data['f1'].max() * 100, color='gray', s=8)
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axs[1].plot(data['Step'], data['-esl'], 'tab:red')
    axs[1].set(xlabel=None)
    axs[1].set_ylabel('-BS')
    # minor_ticks = np.arange(data['-esl'].min(), data['-esl'].max(), 0.02)
    # axs[1].set_yticks(minor_ticks)
    # axs[1].grid(which='major', color='red', linestyle='--', linewidth=0.5, axis="both", alpha=0.5)
    axs[1].axhline(data['-esl'].max(), color='gray', linestyle='dashed', linewidth=0.5)
    axs[1].scatter(data['-esl'].argmax(), data['-esl'].max(), color='gray', s=8)
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    textstr = r'$Cor$=%.2f' % (data['COR_ESL'].max(),)
    props = dict(facecolor='white', alpha=0.6)
    axs[1].text(0.987, 0.05, textstr, transform=axs[1].transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

    axs[2].plot(data['Step'], data['-pesl'], 'tab:green')
    axs[2].set_ylabel('-PBS')
    # minor_ticks = np.arange(data['-pesl'].min(), data['-pesl'].max(), 0.02)
    # axs[2].set_yticks(minor_ticks)
    # axs[2].grid(which='major', color='green', linestyle='--', linewidth=0.5, axis="both", alpha=0.5)'
    axs[2].axhline(data['-pesl'].max(), color='gray', linestyle='dashed', linewidth=0.5)
    axs[2].scatter(data['-pesl'].argmax(), data['-pesl'].max(), color='gray', s=8)
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    textstr = r'$Cor$=%.2f' % (data['COR_PESL'].max(),)
    props = dict(facecolor='white', alpha=0.6)
    axs[2].text(0.987, 0.05, textstr, transform=axs[2].transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # plt.show()
    plt.savefig("history/" + dataset + ".png")
