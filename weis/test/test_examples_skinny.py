import unittest
from weis.test.utils import execute_script

skinny_scripts = [
    "03_NREL5MW_OC3_spar/weis_driver",
    "06_IEA-15-240-RWT/weis_driver",
    "09_design_of_experiments/weis_driver",
]

class TestSkinnyExamples(unittest.TestCase):

    def test_all_scripts(self):
        for k in skinny_scripts:
            with self.subTest(f"Running: {k}", i=k):
                try:
                    execute_script(k)
                    self.assertTrue(True)
                except:
                    self.assertTrue(False)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSkinnyExamples))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
