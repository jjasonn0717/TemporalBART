FROM allennlp/allennlp:v1.1.0

RUN pip install --upgrade pip

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade -r requirements.txt

# Download preprocessed training data from https://drive.google.com/drive/folders/1Cyxi8meJ6TjeayVrnAeerN9FF7pnhyos
RUN gdown "https://drive.google.com/uc?id=1jTvAWKqhokWtVg7IDB64KU1blEjzMcz9"  # KB_arg_overlap_chains_seqdata_scrambled2_train100000.pkl
RUN gdown "https://drive.google.com/uc?id=15kL17NzKbrgCyMHEnfLessoHGsYc_DpK"  # KB_arg_overlap_chains_seqdata_scrambled2_valid.pkl
# CaTeRS evaluation data
RUN gdown "https://drive.google.com/uc?id=1cEjt6Skb0Nnmy2a6XWEL6zVI57idT-Bj"  # caters_entity_chains_seqdata_scrambled2.pkl
# MCTaco evaluation data
RUN gdown "https://drive.google.com/uc?id=1z2V1_YUegKWVXjvaEyR0T-G_4AVAz6Ja"  # mctaco_event_ordering_before_after.json

COPY . .

ENTRYPOINT /bin/bash
