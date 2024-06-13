import unittest
import pickle

class LoaderTestCase(unittest.TestCase):
    def setUp(self):
        self.loader = Loader()  # Create an instance of the Loader class

    def test_from_pickle(self):
        # Define the test data
        data_path = 'test_data.pkl'
        observations = [1, 2, 3]
        dags = [4, 5, 6]
        data = (observations, dags)

        # Save the test data to a pickle file
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

        # Call the from_pickle method
        self.loader.from_pickle(data_path)

        # Check if the observations and dags attributes are correctly set
        self.assertEqual(self.loader.observations, observations)
        self.assertEqual(self.loader.dags, dags)

        # Clean up the test data file
        os.remove(data_path)

if __name__ == '__main__':
    unittest.main()