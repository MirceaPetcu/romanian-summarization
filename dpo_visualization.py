import pickle
import streamlit as st


st.set_page_config(layout="wide")
col1, col2,col3 = st.columns([1, 1, 1])


@st.cache_resource
def get_samples():
    with open('pre_dpo_dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)
    for sample in dataset.values():
        assert len(sample['generated']) == 5, "Wrong number of summaries"

    return dataset

# @st.cache_resource
# def init_dataset():
#     dataset = {i:{'prompt': None, 'chosen': None, 'rejected': None} for i in range(933)}
#     return dataset

@st.cache_resource
def init_dataset():
    with open('dpo_dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)
    return dataset


if 'idx' not in st.session_state:
    st.session_state.idx = 676
if 'document' not in st.session_state:
    st.session_state.document = ''
if 'summaries' not in st.session_state:
    st.session_state.summaries = ''
if 'reference' not in st.session_state:
    st.session_state.reference = ''
if 'chosen_idx' not in st.session_state:
    st.session_state.chosen_idx = ''
if 'rejected_idx' not in st.session_state:
    st.session_state.rejected_idx = ''
if 'original_summaries' not in st.session_state:
    st.session_state.original_summaries = []



def next_item():
    st.session_state.idx = st.session_state.idx + 1
    st.write(st.session_state.idx)


def update_dpo_dataset():
    chosen_idx = int(st.session_state.chosen_idx)
    rejected_idx = int(st.session_state.rejected_idx)
    if -1 < chosen_idx < 6 and -1 < rejected_idx < 6:
        if chosen_idx == 5:
            dpo_dataset[st.session_state.idx]['chosen'] = chosen_idx
        else:
            dpo_dataset[st.session_state.idx]['chosen'] = chosen_idx
        if rejected_idx == 5:
            dpo_dataset[st.session_state.idx]['rejected'] = rejected_idx
        else:
            dpo_dataset[st.session_state.idx]['rejected'] = rejected_idx
        dpo_dataset[st.session_state.idx]['prompt'] = st.session_state.document

        print('\n\n\n')
        print(st.session_state.idx)
        print(dpo_dataset[st.session_state.idx]['prompt'])
        print(dpo_dataset[st.session_state.idx]['chosen'])
        print(dpo_dataset[st.session_state.idx]['rejected'])

        with open('dpo_dataset.pkl', 'wb') as dpo_file:
            pickle.dump(dpo_dataset, dpo_file)
        st.success('DPO dataset updated successfully')
    else:
        st.error('Bad index')

if __name__ == '__main__':
    dataset = get_samples()
    dpo_dataset = init_dataset()
    
    
    with col3:
        if st.button("Next"):
            next_item()
            long_document = dataset[st.session_state.idx]['document']
            summaries = dataset[st.session_state.idx]['generated']
            reference = dataset[st.session_state.idx]['reference']
            st.session_state.original_summaries = summaries
            assert len(st.session_state.original_summaries) == 5, "Wrong number of summaries"
            st.session_state.summaries = ''
            for k,s in enumerate(summaries):
                st.session_state.summaries += f'Summary {k}:\n\n' + s + '\n\n'
            st.session_state.summaries += f'\n\n\nSummary {k+1}:\n\n' + reference
            st.session_state.document = long_document
            st.session_state.reference = reference
           
            del summaries,reference,long_document

    with col1:
        # Display the long document
        st.subheader(f"Document with number {st.session_state.idx}")
        st.write(st.session_state.document)

    with col2:
        # Display summaries with buttons
        st.subheader("Summaries")
        st.write(st.session_state.summaries)

    with col3:
        st.text_input("Index of chosen summary", key='chosen_idx')
        st.text_input("Index of rejected summary", key='rejected_idx')
    with col3:
        if st.button("Update DPO Dataset"):
            update_dpo_dataset()


    


