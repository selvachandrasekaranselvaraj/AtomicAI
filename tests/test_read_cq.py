import unittest
from AtomicAI.io.read_cq import read_cq

class TestReadCq(unittest.TestCase):

	def test_read_cq(self):
		self.assertEqual(list(read_cq('Si.dat').symbols), ['Si']*8)
