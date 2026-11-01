#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines:

	- the `itom_hub_retrofit_impurities_tinyomo` class that extends the class `itom_hub_retrofit_tinyomo`
	  which replaces a Pyomo ABSTRACT and CONCRETE model definition using classes from tinyomo

@author: mathieusa, alexanderkl
"""

__all__ = ('itom_hub_retrofit_impurities_tinyomo',)

from tinyomo import NonNegativeReals, Reals, Set, Param, Var, Constraint
from itom_retrofit_tinyomo import itom_hub_retrofit_tinyomo


class itom_hub_retrofit_impurities_tinyomo(itom_hub_retrofit_tinyomo):
	"""
	Subclass of ITOM Hub Retrofit including constraints on impurity levels (e.g. for steel sector modelling).
	"""

	def __init__(self, InputPath=None, OutputPath=None, config=None):

		super().__init__(InputPath=InputPath, OutputPath=OutputPath, config=config)

		###############
		#    Sets     #
		###############

		# Object containing info on all Sets

		# Do not remove or rename these sets, they are copies of the PRODUCT set
		# to differentiate between "product main" (PRODUCT_1) and "product impurity" (PRODUCT_2)
		self.PRODUCT_1 = Set(SetName='PRODUCT_1', SetsGroup=self.AllSets)
		self.PRODUCT_2 = Set(SetName='PRODUCT_2', SetsGroup=self.AllSets)

		##############
		# Parameters #
		##############

		self.MaxImpurity = Param(self.PRODUCT, self.PRODUCT, default=self.HighMaxDefault,
								 ParamsGroup=self.AllParams, ParamName='MaxImpurity')

		#############
		# Variables #
		#############

		###############
		# Constraints #
		###############

		self.IP1_ImpuritiesInProductsLimit = Constraint(self.LOCATION, self.TECHNOLOGY, self.PRODUCT, self.YEAR,
														rule=self.IP1_ImpuritiesInProductsLimit_rule,
														ConsName='IP1_ImpuritiesInProductsLimit',
														ConsGroup=self.AllCons,
														sparse=self._sparse_impurities if self.sparse_enumeration else None)

	###########
	# METHODS #
	###########

	###############
	# Constraints #
	###############

	#########       	Impurities in products	     	#############

	def _sparse_impurities(self):
		'''
		Index tuples of IP1 (location, technology, product, year), see the note on sparse enumeration in
		itom_hub_tinyomo. The rule returns a constraint only for a product i that the technology produces and
		for which MaxImpurity(p, i) is not the default for some product p. Location and year do not matter.
		If MaxImpurity has no data at all, the block is empty.
		'''
		technologies, products = self._set_values(self.TECHNOLOGY), self._set_values(self.PRODUCT)
		limited = [any(self.MaxImpurity.get_value(p, i) != self.HighMaxDefault for p in products) for i in products]
		products_of_technology = [[ii for ii, i in enumerate(products)
								   if limited[ii] and self.ProductFromTechnology.get_value(t, i) == 1]
								  for t in technologies]
		for il in range(self.LOCATION.len):
			for it in range(len(technologies)):
				for ii in products_of_technology[it]:
					for iy in range(self.YEAR.len):
						yield (il, it, ii, iy)

	def IP1_ImpuritiesInProductsLimit_rule(self, l, t, i, y):
		"""
		*Constraint:* some technologies generate "by-products" that actually represent impurities in the main product
		(e.g. Cu in steel). The parameter MaxImpurity defines the impurity limit (share) tolerated for a given product.
		In this constraint, for technologies where materials of different grades are mixed, the final product's
		impurity content should be be smaller than MaxImpurity rate applied to the production of the main product.
		::

			LocalProductionByTechnology(l,t,i,y) <= MaxImpurity(p,i)* LocalProductionByTechnology(l,t,p,y))

		"""

		if self.ProductFromTechnology.get_value(t, i) == 1:
			RelevantProduct = [p for p in self.PRODUCT.data.VALUE if
							   (p != i) and (self.ProductFromTechnology.get_value(t, p) == 1)]
			if RelevantProduct:
				for p in RelevantProduct:
					MaxImpurity = self.MaxImpurity.get_value(p, i)
					if MaxImpurity == self.HighMaxDefault:
						return None
					else:
						if MaxImpurity != 0:
							lhs = [(1, self.LocalProductionByTechnology.get_index_label(l, t, i, y)),
									(-1 * MaxImpurity,
									self.LocalProductionByTechnology.get_index_label(l, t, p, y))]
							rhs = 0
							sense = '<='
							return {'lhs': lhs, 'rhs': rhs, 'sense': sense}
						else:
							lhs = [(1, self.LocalProductionByTechnology.get_index_label(l, t, i, y))]
							rhs = 0
							sense = '<='
							return {'lhs': lhs, 'rhs': rhs, 'sense': sense}
			else:
				return None
		else:
			return None
