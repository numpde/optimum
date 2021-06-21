# RA, 2021-06-21

import numpy as np
import pandas as pd

from pathlib import Path
from more_itertools import pairwise

from tcga.utils import unlist1, mkdir
from plox import Plox, rcParam

out_dir = mkdir(Path(__file__).with_suffix(''))

df = pd.read_csv(unlist1(Path(__file__).parent.glob("census*/*/data.csv")), header=1)

prefix = "Estimate!!Total!!Population 16 years and over with earnings!!FULL-TIME, YEAR-ROUND WORKERS WITH EARNINGS!!"

df = df[
    [
        c
        for c in df.columns
        if c.startswith(prefix)
    ]
]

df.columns = [c[len(prefix):] for c in df.columns]


brackets = list(pairwise([0, 10000, 15000, 25000, 35000, 50000, 65000, 75000, 100000, 200000]))

df = df.iloc[:, 0:len(brackets)]

df = df.T.rename(columns={0: "number"})
df['%'] = df.number / sum(df.number)

df['bracket'] = brackets

df['hourly'] = [np.mean(bracket) / (52 * 8 * 5) for bracket in df.bracket]
df['secondly'] = df.hourly / (60 * 60)

print(df.to_markdown())


df.to_csv(out_dir / "wages.tsv", sep='\t')


with Plox({rcParam.Figure.figsize: (8, 4), rcParam.Text.usetex: True}) as px:
    brackets = pd.DataFrame(columns=['a', 'b'], data=list(df.bracket))
    brackets['d'] = brackets.b - brackets.a
    brackets['%'] = list(df['%'])

    brackets['density'] = brackets['%'] / brackets.d

    px.a.bar(x=brackets.a, width=brackets.d, height=brackets.density, align='edge', label="census")

    px.a.set_yticks([])

    px.a.set_xlim(0, 1e5)

    # The underlying normal
    sigma = 0.7
    mu = np.round(np.log(6e4))

    from scipy.stats import lognorm
    xx = np.linspace(0, 1e5, 101)
    yy = lognorm(s=sigma, scale=np.exp(mu)).pdf(xx)
    px.a.plot(xx, yy, '--', color='C3', zorder=100, lw=2, label=rf"log-normal $\mu = {mu}$, $\sigma = {sigma}$")

    px.a.set_xlabel("Yearly income (\$), New York City, 2019")
    px.a.set_ylabel("Population 16+ years with earnings")

    px.a.legend()

    px.a.grid(lw=0.3)

    px.f.savefig(out_dir / "wages_hist.png")
