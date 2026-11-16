"""Publication exports of observed prospective results, without synthetic curves."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .run import OUT,BASELINES,ABLATIONS


def export(fig,name):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    releases=[p.name for p in sorted(OUT.glob('r*')) if (p/'metrics.csv').exists() and (p/'confidence_intervals.csv').exists()]
    methods=BASELINES+['M5']
    fig,axes=plt.subplots(len(releases),1,figsize=(12,3.7*len(releases)),squeeze=False)
    for ax,release in zip(axes[:,0],releases):
        d=pd.read_csv(OUT/release/'metrics.csv');m=d.groupby('method').mean(numeric_only=True).loc[methods]
        c=pd.read_csv(OUT/release/'confidence_intervals.csv')
        c=c.loc[c.metric.eq('recall')].set_index('method').loc[methods]
        worst=d.groupby('method').fpr.max()
        colors=['#d87567' if worst[n]>.01 else '#168aad' if n=='M5' else '#9aa2aa' for n in methods]
        x=np.arange(len(methods));ax.bar(x,100*m.recall,color=colors)
        ax.vlines(x,100*c.lower,100*c.upper,color='black',linewidth=1)
        ax.set_xticks(x,methods);ax.set_ylim(0,100);ax.set_ylabel('Recall при фиксированном пороге, %')
        comparator=json.loads((OUT/release/'comparator.json').read_text())['comparator']
        ax.set_title(f'{release}; comparator выбран на validation: {comparator}')
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(handles=[Patch(color='#168aad',label='M5'),Patch(color='#9aa2aa',label='Baseline: все FPR≤1%'),
        Patch(color='#d87567',label='Есть seed с FPR>1%')],fontsize=8,loc='upper right')
    fig.suptitle('Проспективное обнаружение; 95% пользовательский bootstrap CI')
    export(fig,'prospective_recall')
    fig,axes=plt.subplots(len(releases),1,figsize=(12,3.6*len(releases)),squeeze=False)
    for ax,release in zip(axes[:,0],releases):
        d=pd.read_csv(OUT/release/'metrics.csv');x=np.arange(len(methods))
        for offset,seed in zip(np.linspace(-.2,.2,5),range(42,47)):
            m=d.loc[d.seed.eq(seed)].set_index('method').loc[methods]
            ax.scatter(x+offset,100*m.fpr,label=f'seed {seed}',s=20)
        ax.axhline(1,color='#b33131',linestyle='--',label='FPR 1%')
        ax.set_xticks(x,methods);ax.set_ylabel('Фактический FPR, %');ax.set_title(release)
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(ncol=6,fontsize=8)
    fig.suptitle('Пять запусков; пороги не изменялись после validation')
    export(fig,'false_positive_rates')
    fig,axes=plt.subplots(len(releases),2,figsize=(13,3.7*len(releases)),squeeze=False)
    for row,release in enumerate(releases):
        d=pd.read_csv(OUT/release/'history_metrics.csv')
        comparator=json.loads((OUT/release/'comparator.json').read_text())['comparator']
        bins=['0-9','10-29','30-59','60+']
        for method in ['M5',comparator,'AB']:
            m=d.loc[d.method.eq(method)].groupby('history_bin',observed=True).mean(numeric_only=True).reindex(bins)
            axes[row,0].plot(range(4),100*m.recall,marker='o',label=method)
            axes[row,1].plot(range(4),100*m.fpr,marker='o',label=method)
        support=d.loc[d.method.eq('M5')].groupby('history_bin',observed=True).positives.first().reindex(bins).fillna(0)
        for ax in axes[row]:
            ax.set_xticks(range(4),[f'{b}\n+дней: {int(support[b])}' for b in bins])
            ax.set_xlabel('Эффективный объём прошлой истории M5');ax.set_title(release);ax.legend();ax.grid(alpha=.2)
        axes[row,0].set_ylabel('Recall, %');axes[row,0].set_ylim(0,100)
        axes[row,1].set_ylabel('FPR, %');axes[row,1].axhline(1,color='red',linestyle='--')
    fig.suptitle('Одинаковые history-группы наблюдений для сравниваемых методов')
    export(fig,'history_length')
    capacities=[r for r in releases if (OUT/r/'history_capacity_metrics.csv').exists()]
    if capacities:
        fig,axes=plt.subplots(len(capacities),2,figsize=(12,3.6*len(capacities)),squeeze=False)
        for row,release in enumerate(capacities):
            d=pd.read_csv(OUT/release/'history_capacity_metrics.csv')
            for col,metric,label in [(0,'recall','Recall, %'),(1,'fpr','FPR, %')]:
                group=d.groupby('capacity')[metric]
                x=group.mean().index.to_numpy()
                axes[row,col].plot(x,100*group.mean(),marker='o',color='#168aad')
                axes[row,col].fill_between(x,100*group.min(),100*group.max(),alpha=.2,color='#168aad')
                axes[row,col].set_xticks(x);axes[row,col].set_xlabel('Максимум прошлых user-days в личной истории')
                axes[row,col].set_ylabel(label);axes[row,col].set_title(release);axes[row,col].grid(alpha=.2)
            axes[row,0].set_ylim(0,100);axes[row,1].axhline(1,color='red',linestyle='--')
        fig.suptitle('Дополнительная sensitivity-проверка: среднее и диапазон пяти seeds (не CI)')
        export(fig,'history_capacity')
    fig,axes=plt.subplots(len(releases),1,figsize=(12,3.7*len(releases)),squeeze=False)
    for ax,release in zip(axes[:,0],releases):
        d=pd.read_csv(OUT/release/'role_metrics.csv')
        comparator=json.loads((OUT/release/'comparator.json').read_text())['comparator']
        selected=d.loc[d.method.isin(['M5',comparator])].groupby(['role','method']).mean(numeric_only=True)
        worst=selected.fpr.groupby('role').max().nlargest(12).index
        x=np.arange(len(worst))
        for offset,method,color in [(-.18,'M5','#168aad'),(.18,comparator,'#999999')]:
            ax.bar(x+offset,100*selected.xs(method,level='method').loc[worst,'fpr'],width=.36,label=method,color=color)
        ax.set_xticks(x,worst,rotation=35,ha='right');ax.set_ylabel('Ролевой FPR, %');ax.set_title(release)
        ax.axhline(5,color='#b33131',linestyle='--',label='Ориентир 5%');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Роли с наибольшими ошибками; малые роли также включены')
    export(fig,'role_false_positives')
    ablations=['M5']+ABLATIONS
    fig,axes=plt.subplots(len(releases),1,figsize=(12,3.5*len(releases)),squeeze=False)
    for ax,release in zip(axes[:,0],releases):
        d=pd.read_csv(OUT/release/'metrics.csv').groupby('method').mean(numeric_only=True).loc[ablations]
        x=np.arange(len(ablations));ax.bar(x-.18,100*d.recall,width=.36,label='Recall, %',color='#168aad')
        other=ax.twinx();other.plot(x,100*d.fpr,'o-',color='#d36c41',label='FPR, %');other.set_ylabel('FPR, %');other.axhline(1,color='#d36c41',linestyle='--')
        ax.set_xticks(x,ablations,rotation=20,ha='right');ax.set_ylabel('Recall, %');ax.set_ylim(0,100);ax.set_title(release)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Абляции: неизменённый основной M5 и отключение отдельных механизмов')
    export(fig,'ablations')
    pairs=pd.read_csv(OUT/'paired_comparisons.csv')
    fig,ax=plt.subplots(figsize=(9,2+len(releases)))
    for i,release in enumerate(releases):
        row=pairs.loc[pairs.release.eq(release)&pairs.primary&pairs.metric.eq('recall')].iloc[0]
        ax.plot([100*row.lower,100*row.upper],[i,i],color='#168aad')
        ax.scatter(100*row.difference,i,color='#168aad')
    ax.set_yticks(range(len(releases)),releases);ax.axvline(0,color='#777');ax.axvline(5,color='#b33131',linestyle='--',label='Требуемый прирост +5 п.п.')
    ax.set_xlabel('Разность Recall M5 − заранее выбранный baseline, п.п.');ax.legend();ax.grid(axis='x',alpha=.2)
    ax.set_title('95% парные cluster-bootstrap CI; пороги фиксированы')
    export(fig,'paired_recall_difference')
    fig,axes=plt.subplots(len(releases),1,figsize=(12,3.6*len(releases)),squeeze=False)
    for ax,release in zip(axes[:,0],releases):
        d=pd.read_csv(OUT/release/'attack_delays.csv')
        d=d.loc[d.eligible_7day]
        if d.empty:
            ax.text(.5,.5,'Нет новых инцидентов с 7 сутками наблюдения;\nперенесённые через границу test инциденты показаны в CSV',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(release);ax.set_xticks([]);ax.set_yticks([]);continue
        means=d.groupby(['seed','method']).restricted_delay_7day.mean().groupby('method').mean().reindex(methods)
        ax.bar(range(len(methods)),means,color=['#168aad' if m=='M5' else '#999999' for m in methods])
        n=d.loc[d.seed.eq(42),'incident_id'].nunique()
        ax.set_title(f'{release}: новых подходящих инцидентов {n}')
        ax.set_xticks(range(len(methods)),methods);ax.set_ylabel('Ограниченная задержка, сутки');ax.set_ylim(0,7)
    fig.suptitle('min(задержка, 7); необнаруженные получают 7 суток, разрешение — день')
    export(fig,'detection_delay')
    fig,ax=plt.subplots(figsize=(12,5));ax.axis('off')
    boxes=[(.02,.65,'Прошлая история\nuser / role / global'),(.3,.65,'n_eff и устойчивость D\nадаптивные веса'),(.58,.65,'Текущий L\nи затухающий CUSUM'),(.3,.15,'Ролевой сдвиг d\nиз guard-validation'),(.65,.15,'S = max(L, C) − d\nтревога S > θ')]
    for x,y,label in boxes:ax.text(x,y,label,ha='left',va='center',bbox=dict(boxstyle='round,pad=.8',fc='#eef5f8',ec='#168aad'),transform=ax.transAxes)
    for start,end in [((.23,.65),(.28,.65)),((.51,.65),(.56,.65)),((.76,.55),(.76,.28)),((.54,.15),(.63,.15))]:
        ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#333'))
    ax.text(.02,.02,'После score: обновление только личной истории прошлым ограниченным наблюдением.\nРолевой сдвиг и конечный порог заморожены до test; истинные test-метки модели недоступны.',transform=ax.transAxes)
    ax.set_title('M5: причинная адаптация при фиксированном пороге')
    export(fig,'method_diagram')


if __name__=='__main__':main()
