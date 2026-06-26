import zarr

class DataWriter():
    def __init__(self, path="/media/egghead/Scratch/joey/simulation_data/patient_two_data.zarr/", num_batch=10, seq_len=10, n_dim=25):
        self.path = path
        self.num_batch = int(num_batch)
        self.seq_len = int(seq_len)
        self.n_dim = int(n_dim)

        self.batch_idx = 0 

        self.store = zarr.storage.LocalStore(self.path)
        self.root = zarr.group(store=self.store, overwrite=True)

        self.data = self.root.create_array(
            name="episodes",
            shape=(self.num_batch, self.seq_len, self.n_dim),
            chunks=(1, self.seq_len, self.n_dim), 
            dtype="f8",
            overwrite=True
        )

    def __del__(self):
        self.store.close()

    def write_episode(self, data):
        #(seq_len, n_dim)
        self.data[self.batch_idx, :, :] = data
        self.batch_idx += 1 

