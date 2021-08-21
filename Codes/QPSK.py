import numpy as np
import matplotlib.pyplot as plt


def duplicate(arr):
	return np.copy(arr)


def modulateBits(dataCopy):
	QPSK_Bits = (-dataCopy[0:len(dataCopy):2]) + 1j * (-dataCopy[1:len(dataCopy):2])
	return (1 / np.sqrt(2)) * QPSK_Bits


def convertToQPSK(data):
	dataCopy = duplicate(data)
	dataCopy[dataCopy == 0] = -1

	return modulateBits(dataCopy)


def generateNormalNumber(mean , variance , length):
	# loc: Mean , scale: Variance , size: how many numbers
	return np.random.normal(loc=mean, scale=variance , size=length)


def calculateComplexFormatValue(I , Q):
	return (1 / np.sqrt(2)) * (I + 1j * Q)


def deployWirelessGain(QPSK_Bits):
	gain = calculateComplexFormatValue(generateNormalNumber(0 , 1 , int(len(QPSK_Bits))) , generateNormalNumber(0 , 1 , int(len(QPSK_Bits))))
	return gain * QPSK_Bits , gain


def calculateSigma2(SNR):
	return 1 / SNR


def addAWGN(QPSK_Bits , SNR):
	variance = calculateSigma2(SNR)
	noise = calculateComplexFormatValue(generateNormalNumber(0 , calculateSigma2(SNR) , int(len(QPSK_Bits))) , 
										generateNormalNumber(0 , calculateSigma2(SNR) , int(len(QPSK_Bits))) )

	return QPSK_Bits + noise


def removeGainFromReceiverInputSignal(receivedSignal , gain):
	return receivedSignal / gain


def mergeDemodulatedBits(evenQPSK_Bits , oddQPSK_Bits , length):
	decodedSignalLength = length * 2
	decodedSignal = np.zeros(decodedSignalLength, dtype=int)
	decodedSignal[0:decodedSignalLength:2] = evenQPSK_Bits
	decodedSignal[1:decodedSignalLength:2] = oddQPSK_Bits
	return decodedSignal


def demodulateBits(receivedSignalNoGain):
	evenQPSK_Bits = np.real(receivedSignalNoGain); oddQPSK_Bits = np.imag(receivedSignalNoGain);
	evenQPSK_Bits[evenQPSK_Bits > 0] = 0; evenQPSK_Bits[evenQPSK_Bits < 0] = 1
	oddQPSK_Bits[oddQPSK_Bits > 0] = 0; oddQPSK_Bits[oddQPSK_Bits < 0] = 1
	return evenQPSK_Bits , oddQPSK_Bits


def generateDecodedSignal(receivedSignalNoGain):
	CopyQPSK_Bits = duplicate(receivedSignalNoGain)
	CopyQPSK_Bits *= np.sqrt(2)

	evenQPSK_Bits , oddQPSK_Bits = demodulateBits(CopyQPSK_Bits)
	return mergeDemodulatedBits(evenQPSK_Bits , oddQPSK_Bits , len(receivedSignalNoGain))


def generateTransmitterOutputSignal(data):
	channelSignal , gain = deployWirelessGain(convertToQPSK(data))
	return channelSignal , gain

def generateReceiverInputSignal(channelSignal , SNR):
	return addAWGN(channelSignal , SNR)


def QPSK(data , SNR , showPlots=True):
	# Transmitter Side
	channelSignal , gain = generateTransmitterOutputSignal(data)

	# Receiver Side
	receivedSignal = generateReceiverInputSignal(channelSignal , SNR)
	receivedSignalNoGain = removeGainFromReceiverInputSignal(receivedSignal , gain)
	decodedSignal = generateDecodedSignal(receivedSignalNoGain)

	if(showPlots):
		transmitted = convertToQPSK(data)

		plt.scatter(receivedSignalNoGain.real, receivedSignalNoGain.imag, label="received", color="blue", s=10)
		plt.scatter(transmitted.real, transmitted.imag, label="transmitted", color="red", s=50)
		plt.xlim(-3, 3); plt.ylim(-3, 3)
		plt.axhline(0, color="black"); plt.axvline(0, color="black")
		plt.xlabel("I"); plt.ylabel("Q")
		plt.title("QPSK with SNR: " + str(SNR))
		plt.legend()
		plt.show()
	
	else:
		numMismatches = 0
		for i in range(len(data)):
			if(decodedSignal[i] != data[i]):
				numMismatches += 1

		return (numMismatches / len(data)) * 100



QPSK(np.random.randint(0, 2, 5000) , 0.1)
QPSK(np.random.randint(0, 2, 5000) , 1)
QPSK(np.random.randint(0, 2, 5000) , 10)

mismatches = []
for SNR in np.arange(0.1, 10, 0.1):
	mismatches.append(QPSK(np.random.randint(0, 2, 5000) , SNR , False))

plt.scatter(np.arange(0.1, 10, 0.1) , mismatches)
plt.xlabel("SNR")
plt.ylabel("Probability Percentage")
plt.title("Average Probability of Mismatch With Respect to SNR")
plt.show()
 