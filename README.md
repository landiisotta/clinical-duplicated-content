# Clinical Text Deduplication

Clinical notes are a unique source of information on patients' status and disease progression. Although large language 
models (LMs) have proven to learn medical knowledge, to some extent, they still require specialized domain
adaptation for improved downstream clinical tasks. 

Clinical notes are characterized by high levels of duplicated content and the extent to which such duplicates impact 
LMs and downstream tasks is still understudied.

This work includes:

1. The identification of duplicates defined as: _witin-note_; _between-note not relevant_; _between-note relevant_;
    - Clinical relevance is identified via a fine-tuned GatorTron model that classifies duplicates 
    as clinically relevant or not relevant.
    
2. Text deduplication: three possible configurations available. WN = clinical text w/o within-note duplicates;
WNNR = clinical text w/o within-note and between-note not relevant duplicates; WNBN = clinical text with all duplicates
dropped;

3. Investigation of the impact of deduplication practices on LMs via PPL;

4. Investigation of the impact of deduplication practices on downstream tasks via prompt-based learning.
 
By leveraging large real-world clinical corpora, we showed that such large datasets are characterized by high levels
 of duplication. We argue that clinical text deduplication, based on clinical relevance and duplicates distribution, 
 can help clinical LMs encode less redundant information in a more efficient manner and improve classification tasks
  via prompt-based learning.

**Remark**: the original work was performed on three clinical datasets: MIMIC-III v1.4, n2c2 challenge datasets, 
and ICU notes from the Mount Sinai Health System Data Warehouse (NYC). 
Access to such datasets is restricted, hence, we provide here a subset of the English Wikipedia dataset[^1] 
that was rearranged to resemble the structure of clinical corpora and can be used to test the code.

[^1]
Wikipedia data was downloaded as

```
from datasets import load_dataset
wiki = load_dataset("wikipedia", "20220301.en")
```

We selected 1,000 wikipedia documents for dataset 1 and 500 for dataset 2.

   


  
