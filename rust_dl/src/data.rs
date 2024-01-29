use std::{
    fs::File,
    io::BufReader,
};

use noodles::{bed, core as nc, fasta};
use std::path::PathBuf;

use ai_dataloader::{Dataset, GetSample, Len};

struct FastaDataset {
    reader: fasta::IndexedReader<BufReader<File>>,
    records: Vec<bed::Record<3>>,
}

impl FastaDataset {
    fn new(fasta: PathBuf, bed: PathBuf) -> Self {
        let mut fai = fasta.clone();
        fai.push(".fai");
        let index = fasta::fai::read(fai).expect("Error opening FASTA index");
        let reader = File::open(fasta)
            .map(BufReader::new)
            .map(|r| fasta::IndexedReader::new(r, index))
            .expect("Error opening FASTA");

        let mut bed_reader = File::open(bed)
            .map(BufReader::new)
            .map(bed::Reader::new)
            .expect("Error opening BED");

        let records = bed_reader
            .records::<3>()
            .enumerate()
            .map(|(i, r)| r.expect(format!("Malformed BED record {i}").as_str()))
            .collect::<Vec<_>>();

        Self {
            reader,
            records,
        }
    }
}

impl Dataset for FastaDataset {}

impl Len for FastaDataset {
    fn len(&self) -> usize {
        self.records.len()
    }
}

impl GetSample for FastaDataset {
    type Sample = Vec<u8>;

    fn get_sample(&self, index: usize) -> Self::Sample {
        let record = &self.records[index];
        let chrom = record.reference_sequence_name();
        let start = record.start_position();
        let end = record.end_position();

        let region = nc::Region::new(chrom, start..=end);

        let sequence = self
            .reader
            .query(&region)
            .expect("Error querying FASTA");

        sequence.sequence().as_ref().to_vec()
    }
}
