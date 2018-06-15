


# create a mlfm force model which is going
# to be fitted by the NS one step method
mlfm = MLFM(mmats, method="ns_onestep")

##
# mlfm.fit(times, Y)
#
# mlfm.fit will call
#
# mlfm.init 
