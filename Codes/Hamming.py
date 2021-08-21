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


def encodeHamming(data):
	codeGeneratorMatrix = ["1101", "1011", "1000", "0111", "0100", "0010" , "0001"]
	encodedSignal = []

	for i in range(0 , len(data) // 4):
		encoded_7bit = []
		data_4bitSelected = data[i * 4 : (i + 1) * 4]
		for matCode in codeGeneratorMatrix:
			row = list(map(int , matCode))
			encoded_7bit.append(np.dot(row , data_4bitSelected) % 2)

		encodedSignal.extend(encoded_7bit)

	return encodedSignal


def decodeHamming(hammingDecodedSignal):
	parityCheckMatrix = ["1010101" , "0110011" , "0001111"]
	codeDecoderMatrix = ["0010000" , "0000100" , "0000010" , "0000001"]
	decodedSignal = []

	for i in range(0 , len(hammingDecodedSignal) // 7):
		syndromeVectorValue = ""
		signal_7bitSelected = hammingDecodedSignal[i * 7 : (i + 1) * 7]
		
		for matCode in parityCheckMatrix:
			row = list(map(int , matCode))
			syndromeVectorValue += str(np.dot(row , signal_7bitSelected) % 2)

		if(int(syndromeVectorValue , 2) != 0):
			signal_7bitSelected[int(syndromeVectorValue , 2) - 1] = 1 - signal_7bitSelected[int(syndromeVectorValue , 2) - 1]

		for matCode in codeDecoderMatrix:
			row = list(map(int , matCode))
			decodedSignal.extend(list(map(int , bin(np.dot(row , signal_7bitSelected)).replace("0b", ""))))

	return decodedSignal



def Hamming7_4(data , SNR , showPlots=True):
	encodedSignal = encodeHamming(data)
	# Transmitter Side
	channelSignal , gain = generateTransmitterOutputSignal(encodedSignal)

	# Receiver Side
	receivedSignal = generateReceiverInputSignal(channelSignal , SNR)
	receivedSignalNoGain = removeGainFromReceiverInputSignal(receivedSignal , gain)
	hammingDecodedSignal = generateDecodedSignal(receivedSignalNoGain)
	decodedSignal = decodeHamming(hammingDecodedSignal)

	if(showPlots):
		transmitted = convertToQPSK(encodedSignal)

		plt.scatter(receivedSignalNoGain.real, receivedSignalNoGain.imag, label="received", color="blue", s=10)
		plt.scatter(transmitted.real, transmitted.imag, label="transmitted", color="red", s=50)
		plt.xlim(-3, 3); plt.ylim(-3, 3)
		plt.axhline(0, color="black"); plt.axvline(0, color="black")
		plt.xlabel("I"); plt.ylabel("Q")
		plt.title("(7,4) Hamming with SNR: " + str(SNR))
		plt.legend()
		plt.show()
	
	else:
		numMismatches = 0
		for i in range(len(data)):
			if(decodedSignal[i] != data[i]):
				numMismatches += 1

		return (numMismatches / len(data)) * 100



Hamming7_4(np.random.randint(0, 2, 5000) , 0.1)
Hamming7_4(np.random.randint(0, 2, 5000) , 1)
Hamming7_4(np.random.randint(0, 2, 5000) , 10)

mismatches = []
for SNR in np.arange(0.1, 10, 0.1):
	mismatches.append(Hamming7_4(np.random.randint(0, 2, 5000) , SNR , False))

plt.scatter(np.arange(0.1, 10, 0.1) , mismatches)
plt.xlabel("SNR")
plt.ylabel("Probability Percentage")
plt.title("Average Probability of Mismatch With Respect to SNR")
plt.show()
 