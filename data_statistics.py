import numpy as np
import json

class Stat:
	data_stat = {}
	origin_data = None
	def add(self, dict, id, key):
		if id not in dict:
			dict[id] = {}
		if key not in dict[id]:
			dict[id][key] = 1
		else:
			dict[id][key] += 1;

	def load_data(self, path):
		self.origin_data = np.loadtxt(path, dtype=np.str, delimiter=",")

	def stat(self, data):
		users = {}
		merchants = {}
		coupons = {}
		for line in data:
			uid = line[0]
			mid = line[1]
			cid = line[2]
			discount_rate = line[3]
			if len(line) > 6:
				date = line[6]
			else:
				date = "null"
			
			# coupon
			if cid != "null":
				self.add(users, uid, "num_coupons")
				self.add(merchants, mid, "num_coupons")
				self.add(coupons, cid, "num_coupons")
				coupons[cid]["discount_rate"] = discount_rate
			
			# consumes
			if date != "null":
				self.add(users, uid, "num_consumes")
				self.add(merchants, mid, "num_consumes")
				self.add(coupons, cid, "num_consumes")
				
			# coupon and consumes
			if cid != "null" and date != "null":
				self.add(users, uid, "num_coupon_consumes")
				self.add(coupons, cid, "num_coupon_consumes")	
				self.add(merchants, mid, "num_coupon_consumes")	
				
			self.data_stat["users"] = users
			self.data_stat["merchants"] = merchants
			self.data_stat["coupons"] = coupons
				
if __name__ == "__main__":
	#path = "ccf_offline_stage1_train.csv"
	path = "head100"
	data_stat = Stat()
	data_stat.load_data(path)
	data_stat.stat(data_stat.origin_data)
	
	print "users:"
	f_users = open("users", "w")
	users = data_stat.data_stat["users"]
	for key in users:
		print >> f_users, key + "\t" + json.dumps(users[key])
	f_users.close()
		
	print "merchants:"
	merchants = data_stat.data_stat["merchants"]
	f_merchants = open("merchants", "w")
	for key in merchants:
		print >> f_merchants,key + "\t" + json.dumps(merchants[key])
	f_merchants.close()
	
	print "coupons:"
	f_coupons = open("coupons", "w")
	coupons = data_stat.data_stat["coupons"]
	for key in coupons:
		print >> f_coupons,key + "\t" + json.dumps(coupons[key])
	f_coupons.close()
	
