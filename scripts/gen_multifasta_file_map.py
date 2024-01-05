from pathlib import Path

import typer


def main(fastas_dir: Path, bed_dir: Path, seqlen: int, out: Path):
    from itertools import chain

    import polars as pl
    
    fastas = list(chain((fastas_dir / 'mammals').glob('*.fna.gz'), (fastas_dir / 'primates').glob('*.fna.gz')))
    beds = []
    for fasta in fastas:
        beds.append(bed_dir / fasta.parent.name / f'{fasta.name}_windows_{seqlen}.bed')
    
    df = pl.DataFrame({'fasta': [str(p) for p in fastas], 'bed': [str(p) for p in beds]})
    separator = ',' if out.suffix == '.csv' else '\t'
    df.write_csv(out, separator=separator)


if __name__ == "__main__":
    typer.run(main)