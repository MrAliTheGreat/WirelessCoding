import numpy as np
import matplotlib.pyplot as plt


def duplicate(arr):
	return np.copy(arr)


def separateEvenOdd(arr , limit):
	return arr[0:limit:2] , arr[1:limit:2]


def modulateBits(dataCopy):
	evenDataCopy , oddDataCopy = separateEvenOdd(dataCopy , len(dataCopy))
	newData = 2 * evenDataCopy + oddDataCopy

	newDataCopy = duplicate(newData)
	newData[newDataCopy == 3] = 1; newData[newDataCopy == 1] = 3

	evenNewData , oddNewData = separateEvenOdd(newData , len(dataCopy) // 2)

	return (1 / np.sqrt(10)) * (evenNewData - 1j * oddNewData)


def convertTo16QAM(data):
	dataCopy = duplicate(data)
	dataCopy[dataCopy == 0] = -1

	return modulateBits(dataCopy)


def generateNormalNumber(mean , variance , length):
	# loc: Mean , scale: Variance , size: how many numbers
	return np.random.normal(loc=mean, scale=variance , size=length)


def calculateComplexFormatValue(I , Q):
	return (1 / np.sqrt(2)) * (I + 1j * Q)


def deployWirelessGain(QAM_Bits):
	gain = calculateComplexFormatValue(generateNormalNumber(0 , 1 , int(len(QAM_Bits))) , generateNormalNumber(0 , 1 , int(len(QAM_Bits))))
	return gain * QAM_Bits , gain


def calculateSigma2(SNR):
	return 1 / SNR


def addAWGN(QAM_Bits , SNR):
	variance = calculateSigma2(SNR)
	noise = calculateComplexFormatValue(generateNormalNumber(0 , calculateSigma2(SNR) , int(len(QAM_Bits))) , 
										generateNormalNumber(0 , calculateSigma2(SNR) , int(len(QAM_Bits))) )

	return QAM_Bits + noise


def removeGainFromReceiverInputSignal(receivedSignal , gain):
	return receivedSignal / gain


def mergeDemodulatedBits(evenQAM_Bits , oddQAM_Bits , length):
	decodedSignalLength = length * 2
	decodedSignal = np.zeros(decodedSignalLength, dtype=int)
	decodedSignal[0:decodedSignalLength:2] = evenQAM_Bits
	decodedSignal[1:decodedSignalLength:2] = oddQAM_Bits
	return decodedSignal


def demodulateBits(receivedSignalNoGain):
	even16QAM_Bits = np.real(receivedSignalNoGain); odd16QAM_Bits = -np.imag(receivedSignalNoGain);
	evenCopy = duplicate(even16QAM_Bits); oddCopy = duplicate(odd16QAM_Bits)
	
	even16QAM_Bits[evenCopy > 2] = 3; even16QAM_Bits[evenCopy < 2] = 1; even16QAM_Bits[evenCopy < 0] = -1; even16QAM_Bits[evenCopy < -2] = -3
	odd16QAM_Bits[oddCopy > 2] = 3; odd16QAM_Bits[oddCopy < 2] = 1; odd16QAM_Bits[oddCopy < 0] = -1; odd16QAM_Bits[oddCopy < -2] = -3
	return even16QAM_Bits , odd16QAM_Bits


def generateDecodedSignal(receivedSignalNoGain):
	CopyQAM_Bits = duplicate(receivedSignalNoGain)
	CopyQAM_Bits *= np.sqrt(10)

	evenQAM_Bits , oddQAM_Bits = demodulateBits(CopyQAM_Bits)
	mergedSignal = mergeDemodulatedBits(evenQAM_Bits , oddQAM_Bits , len(receivedSignalNoGain))

	decodedSignal = np.zeros(len(receivedSignalNoGain) * 2, dtype=int) + 1j * np.zeros(len(receivedSignalNoGain) * 2, dtype=int)
	decodedSignal[mergedSignal == 3] = 1 - 0j; decodedSignal[mergedSignal == 1] = 1 + 1j
	decodedSignal[mergedSignal == -1] = - 0 + 1j; decodedSignal[mergedSignal == -3] = - 0 - 0j

	return mergeDemodulatedBits(np.real(decodedSignal) , np.imag(decodedSignal) , len(receivedSignalNoGain) * 2)


def generateTransmitterOutputSignal(data):
	channelSignal , gain = deployWirelessGain(convertTo16QAM(data))
	return channelSignal , gain

def generateReceiverInputSignal(channelSignal , SNR):
	return addAWGN(channelSignal , SNR)


def _16QAM(data , SNR , showPlots=True):
	# Transmitter Side
	channelSignal , gain = generateTransmitterOutputSignal(data)

	# Receiver Side
	receivedSignal = generateReceiverInputSignal(channelSignal , SNR)
	receivedSignalNoGain = removeGainFromReceiverInputSignal(receivedSignal , gain)
	decodedSignal = generateDecodedSignal(receivedSignalNoGain)

	if(showPlots):
		transmitted = convertTo16QAM(data)

		plt.scatter(receivedSignalNoGain.real, receivedSignalNoGain.imag, label="received", color="blue", s=10)
		plt.scatter(transmitted.real, transmitted.imag, label="transmitted", color="red", s=50)
		plt.xlim(-3, 3); plt.ylim(-3, 3)
		plt.axhline(0, color="black"); plt.axvline(0, color="black")
		plt.xlabel("I"); plt.ylabel("Q")
		plt.title("16QAM with SNR: " + str(SNR))
		plt.legend()
		plt.show()
	
	else:
		numMismatches = 0
		for i in range(len(data)):
			if(decodedSignal[i] != data[i]):
				numMismatches += 1

		return (numMismatches / len(data)) * 100



_16QAM(np.random.randint(0, 2, 5000) , 0.1)
_16QAM(np.random.randint(0, 2, 5000) , 1)
_16QAM(np.random.randint(0, 2, 5000) , 10)

mismatches = []
for SNR in np.arange(0.1, 10, 0.1):
	mismatches.append(_16QAM(np.random.randint(0, 2, 5000) , SNR , False))

plt.scatter(np.arange(0.1, 10, 0.1) , mismatches)
plt.xlabel("SNR")
plt.ylabel("Probability Percentage")
plt.title("Average Probability of Mismatch With Respect to SNR")
plt.show()
 