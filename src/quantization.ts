/*
import { getNumberParts } from './checksum'; // imports function from checksum.ts that converts a number to its IEEE 754 representation

class FloatingPointQuantizer {
*/
/*
    constructor() {
      // Define precision limits for different floating point formats

      this.formats = {
        fp16: {
          mantissa: 10,
          exponent: 5,
          bias: 15,
          maxExponent: 31,
          minExponent: 0
        },
        fp32: {
          mantissa: 23,
          exponent: 8,
          bias: 127,
          maxExponent: 255,
          minExponent: 0
        }
      };
    }
*/ 
/*
    // Clamp function from Equation 13
    clamp(x, alpha, beta) {
      return Math.max(Math.min(x, beta), alpha);
    }
  
    // Rounding function f(·)
    round(x) {
      return Math.round(x);
    }
  
    // Min-Max quantization method (Equation 14)
    minMaxQuantization(data, targetBits = 8) {
      if (!Array.isArray(data) || data.length === 0) {
        throw new Error('Data must be a non-empty array');
      }
  
      const m = Math.min(...data); // minimum value
      const M = Math.max(...data); // maximum value
      const n = Math.pow(2, targetBits); // maximum representable number
  
      // Calculate scale and zero-point
      const s = (n - 1) / (M - m);
      const z = m * (1 - n) / (M - m);
  
      // Apply quantization using Equation 12
      return data.map(x => {
        const clamped = this.clamp(x, m, M);
        return (this.round(s * clamped + z)) - z / s;
      });
    }
  
    // Max-Abs quantization method (Equation 15)
    maxAbsQuantization(data, targetBits = 8) {
      if (!Array.isArray(data) || data.length === 0) {
        throw new Error('Data must be a non-empty array');
      }
  
      const R = Math.max(...data.map(x => Math.abs(x))); // max absolute value
      const n = Math.pow(2, targetBits);
      const M = R; // symmetrical bound
  
      // Calculate scale (zero-point is 0 for symmetrical quantization)
      const s = (n - 1) / R;
      const z = 0;
  
      // Apply quantization using Equation 12
      return data.map(x => {
        const clamped = this.clamp(x, -M, M);
        return this.round(s * clamped + z) / s;
      });
    }
  
    // Convert FP64 to FP32
    quantizeFP64toFP32(value) {
      // Use Float32Array to automatically handle FP32 precision limits
      const float32Array = new Float32Array(1);
      float32Array[0] = value;
      return float32Array[0];
    }
  
    // Convert FP64 to FP16 (IEEE 754 half precision)
    quantizeFP64toFP16(value) {
      if (value === 0) return 0;
      if (!isFinite(value)) {
        return value > 0 ? 0x7C00 : 0xFC00; // +Inf or -Inf
      }
      if (isNaN(value)) {
        return 0x7E00; // NaN
      }
  
      const sign = value < 0 ? 1 : 0;
      const absValue = Math.abs(value);
  
      // Handle special cases
      if (absValue < Math.pow(2, -24)) { // 2^-24 is the smallest positive number in FP16
        return sign << 15; // Zero or negative zero
      }
  
      if (absValue >= 65504) {
        return (sign << 15) | 0x7C00; // Infinity
      }
  
      // Convert to FP16
      let exponent = Math.floor(Math.log2(absValue));
      let mantissa = absValue / Math.pow(2, exponent) - 1;
  
      // Adjust for FP16 bias
      exponent += 15;
  
      if (exponent <= 0) {
        // Subnormal numbers
        mantissa = absValue / Math.pow(2, -14); // scale by minimum normal exponent (1-15)
        exponent = 0;
      } else if (exponent >= 31) {
        // Infinity for overflow
        return (sign << 15) | 0x7C00; // ensures correct sign
      }
  
      // Quantize mantissa to 10 bits
      mantissa = Math.round(mantissa * 1024) & 0x3FF;
  
      return (sign << 15) | (exponent << 10) | mantissa; // << shifts bits to the left
    }
  
    // Convert FP16 back to FP64 for verification
    fp16ToFP64(fp16Value) {
      const sign = (fp16Value >> 15) & 1;
      const exponent = (fp16Value >> 10) & 0x1F;
      const mantissa = fp16Value & 0x3FF;
  
      if (exponent === 0) {
        if (mantissa === 0) {
          return sign ? -0 : 0;
        }
        // Subnormal
        return (sign ? -1 : 1) * (mantissa / 1024) * Math.pow(2, -14);
      }
  
      if (exponent === 31) {
        return mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
      }
  
      // Normal number
      return (sign ? -1 : 1) * (1 + mantissa / 1024) * Math.pow(2, exponent - 15);
    }

    quantizeFP64toFP8_E4M3(value) { // 4 exponent, 3 mantissa bits
        if (value === 0) return 0;
        
        // E4M3 special values
        if (!isFinite(value)) {
          // E4M3 doesn't have infinity - use max finite value with correct sign
          return value > 0 ? 0x7F : 0xFF; // Max positive/negative finite
        }
        if (isNaN(value)) {
          return 0x7F; // NaN representation
        }
      
        const sign = value < 0 ? 1 : 0;
        const absValue = Math.abs(value);
      
        // E4M3 range: smallest subnormal is 2^-9, largest finite is 448
        if (absValue < Math.pow(2, -9)) {
          return sign << 7; // Underflow to zero
        }
      
        if (absValue >= 448) {
          return (sign << 7) | 0x7F; // Overflow to max finite
        }
      
        // Convert to FP8 E4M3
        let exponent = Math.floor(Math.log2(absValue));
        let mantissa = absValue / Math.pow(2, exponent) - 1;
      
        // Adjust for E4M3 bias (7)
        exponent += 7;
      
        if (exponent <= 0) {
          // Subnormal numbers
          // Scale mantissa for subnormal representation
          mantissa = absValue / Math.pow(2, -6); // 2^(1-7) = 2^-6
          exponent = 0;
        } else if (exponent >= 15) {
          // Max finite value in E4M3
          return (sign << 7) | 0x7F;
        }
      
        // Quantize mantissa to 3 bits
        mantissa = Math.round(mantissa * 8) & 0x7; // 8 = 2^3, 0x7 = 111 binary
      
        return (sign << 7) | (exponent << 3) | mantissa;
      }
      
      
    // Usage example
    testFP8Quantization() {
        const testValues = [0, 1, -1, 3.14, 100, 0.001, 1000];
        
        console.log("FP64 -> FP8 E4M3 Quantization:");
        testValues.forEach(val => {
            const quantized = this.quantizeFP64toFP8_E4M3(val);
            console.log(`${val} -> 0x${quantized.toString(16).padStart(2, '0').toUpperCase()}`);

        });
    }
  
    // Batch quantization for arrays
    quantizeArray(data, sourceFormat = 'fp64', targetFormat = 'fp32') {
      if (!Array.isArray(data)) {
        throw new Error('Input must be an array');
      }
  
      return data.map(value => {
        switch (targetFormat.toLowerCase()) {
          case 'fp32':
            return this.quantizeFP64toFP32(value);
          case 'fp16':
            return this.quantizeFP64toFP16(value);
          default:
            throw new Error('Unsupported target format');
        }
      });
    }
  
    // Demonstration method
    demonstrateQuantization() {
      console.log('=== Floating Point Quantization Demo ===\n');
  
      // Test data
      const testData = [3.14159265359, -2.71828182846, 1.41421356237, 0.57721566490, -1.61803398875];
  
      console.log('Original FP64 values:', testData);
  
      // FP32 quantization
      const fp32Results = this.quantizeArray(testData, 'fp64', 'fp32');
      console.log('FP32 quantized:', fp32Results);
  
      // FP16 quantization (showing as hex)
      const fp16Results = testData.map(val => this.quantizeFP64toFP16(val));
      console.log('FP16 quantized (hex):', fp16Results.map(val => '0x' + val.toString(16).toUpperCase()));
  
      // Convert back from FP16 to verify
      const fp16Decoded = fp16Results.map(val => this.fp16ToFP64(val));
      console.log('FP16 decoded back:', fp16Decoded);

      // 8 bit Min-Max quantization example
      console.log('\n=== Min-Max Quantization (8-bit) ===');
      const minMaxResult = this.minMaxQuantization(testData, 8);
      console.log('Min-Max quantized:', minMaxResult);

      // 8 bit Max-Abs quantization example
      console.log('\n=== Max-Abs Quantization (8-bit) ===');
      const maxAbsResult = this.maxAbsQuantization(testData, 8);
      console.log('Max-Abs quantized:', maxAbsResult);

      // 8 bit Min-Max quantization example
      console.log('\n=== Min-Max Quantization (16-bit) ===');
      const minMaxResult16 = this.minMaxQuantization(testData, 16);
      console.log('Min-Max quantized:', minMaxResult16);

      // 8 bit Max-Abs quantization example
      console.log('\n=== Max-Abs Quantization 16-bit) ===');
      const maxAbsResult16 = this.maxAbsQuantization(testData, 16);
      console.log('Max-Abs quantized:', maxAbsResult16);

      // Calculate quantization errors
      console.log('\n=== Quantization Errors ===');
      const fp32Errors = testData.map((orig, i) => Math.abs(orig - fp32Results[i]));
      const fp16ManualErrors = testData.map((orig, i) => Math.abs(orig - fp16Decoded[i]));

      const fp16MaxAbsErrors = testData.map((orig, i) => Math.abs(orig - maxAbsResult16[i]));
      const fp16MinMaxErrors = testData.map((orig, i) => Math.abs(orig - minMaxResult16[i]));
    
      console.log('FP16 MaxAbs errors:', fp16MaxAbsErrors);
      console.log('FP16 MinMax errors:', fp16MinMaxErrors);
      console.log('FP32 errors:', fp32Errors);
      console.log('FP16 errors:', fp16ManualErrors);

      console.log('Average FP16 MaxAbs error:', fp16MaxAbsErrors.reduce((a, b) => a + b) / fp16MaxAbsErrors.length);
      console.log('Average FP16 MinMax error:', fp16MinMaxErrors.reduce((a, b) => a + b) / fp16MinMaxErrors.length);
      console.log('Average FP32 error:', fp32Errors.reduce((a, b) => a + b) / fp32Errors.length);
      console.log('Average FP16 error:', fp16ManualErrors.reduce((a, b) => a + b) / fp16ManualErrors.length);
    }
  }
  
  // Usage example
  const quantizer = new FloatingPointQuantizer();
  
  // Run demonstration
  quantizer.demonstrateQuantization();
  
  // Example usage for individual values
  console.log('\n=== Individual Value Examples ===');
  const testValue = 3.14159265359;
  console.log(`Original: ${testValue}`);
  console.log(`FP32: ${quantizer.quantizeFP64toFP32(testValue)}`);
  console.log(`FP16 (hex): 0x${quantizer.quantizeFP64toFP16(testValue).toString(16).toUpperCase()}`);
  console.log(`FP16 decoded: ${quantizer.fp16ToFP64(quantizer.quantizeFP64toFP16(testValue))}`);
  
  
  const testValue2 = -2.71828182846;
  console.log(`Original: ${testValue2}`);
  console.log(`FP32: ${quantizer.quantizeFP64toFP32(testValue2)}`);
  console.log(`FP16 (hex): 0x${quantizer.quantizeFP64toFP16(testValue2).toString(16).toUpperCase()}`);
  console.log(`FP16 decoded: ${quantizer.fp16ToFP64(quantizer.quantizeFP64toFP16(testValue2))}`);
  
  // Export for use in other modules
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = FloatingPointQuantizer;
  }


function directFP32Conversion(number: number): number {
    const buffer = new ArrayBuffer(4);
    const float32 = new Float32Array(buffer);
    float32[0] = number;
    return float32[0];
}
console.log(directFP32Conversion(3.14159));
*/



/*
interface QuantizationConfig {
    exponentBits: number;
    mantissaBits: number;
    bias: number;
}


function quantizeNumber(
    number: number, 
    config: QuantizationConfig
): number {
    let parts = getNumberParts(number);
    
    // Debug logging
    console.log(`\nDebug for number ${number}:`);
    console.log(`Parts:`, parts);
    console.log(`Exponent: ${parts.exponent}`);
    console.log(`Mantissa: ${parts.mantissa}`);
    
    // Handle special cases
    if (isNaN(number)) return NaN;
    if (number === Infinity) return Infinity;
    if (number === -Infinity) return -Infinity;
    
    // Convert exponent to new biased exponent
    // getNumberParts already returns the unbiased exponent
    let newExponent = parts.exponent + config.bias;
    console.log(`New biased exponent: ${newExponent} (${parts.exponent} + ${config.bias})`);
    
    // Check for overflow/underflow
    let maxExponent = (1 << (config.exponentBits - 1)) - 1;
    console.log(`Max exponent for ${config.exponentBits} bits: ${maxExponent}`);
    console.log(`Exponent range: -${maxExponent} to ${maxExponent}`);
    
    if (newExponent > maxExponent) {
        console.log(`Overflow: ${newExponent} > ${maxExponent}`);
        return parts.sign ? -Infinity : Infinity;
    }
    if (newExponent < -maxExponent) {
        console.log(`Underflow: ${newExponent} < -${maxExponent}`);
        return 0;
    }
    
    // Calculate new mantissa
    let mantissa = parts.mantissa;
    // Scale mantissa to new precision and round
    mantissa = Math.round(mantissa * Math.pow(2, config.mantissaBits - 52));
    console.log(`Scaled mantissa: ${mantissa}`);
    
    // Reconstruct the number using the new exponent (minus bias)
    let result = (parts.sign ? -1 : 1) * 
                 (1 + mantissa / Math.pow(2, config.mantissaBits)) * 
                 Math.pow(2, newExponent - config.bias);
    console.log(`Final calculation: ${parts.sign ? '-' : '+'}1.${mantissa} × 2^${newExponent - config.bias}`);
    console.log(`Result: ${result}`);
                 
    return result;
}

// Example configurations
const FP64_CONFIG: QuantizationConfig = {
    exponentBits: 11,
    mantissaBits: 52,
    bias: 1023
};

const FP32_CONFIG: QuantizationConfig = {
    exponentBits: 8,
    mantissaBits: 23,
    bias: 127
};

const FP16_CONFIG: QuantizationConfig = {
    exponentBits: 5,
    mantissaBits: 10,
    bias: 15
};

const FP8_CONFIG: QuantizationConfig = {
    exponentBits: 4,
    mantissaBits: 3,
    bias: 7
};

// Test with different numbers
const testNumbers = [3.14159, 1.0, 0.1, 1000.0, -42.5];

testNumbers.forEach(number => {
    console.log(`\nTesting number: ${number}`);
    let fp64 = quantizeNumber(number, FP64_CONFIG);
    let fp32 = quantizeNumber(number, FP32_CONFIG);
    let fp16 = quantizeNumber(number, FP16_CONFIG);
    let fp8 = quantizeNumber(number, FP8_CONFIG);

    console.log(`Original: ${number}`);
    console.log(`FP64: ${fp64}`);
    console.log(`FP32: ${fp32}`);
    console.log(`FP16: ${fp16}`);
    console.log(`FP8: ${fp8}`);
    
    // Show differences
    console.log('Differences from original:');
    console.log(`FP64 diff: ${Math.abs(number - fp64)}`);
    console.log(`FP32 diff: ${Math.abs(number - fp32)}`);
    console.log(`FP16 diff: ${Math.abs(number - fp16)}`);
    console.log(`FP8 diff: ${Math.abs(number - fp8)}`);
});

*/
// Alternative: Direct conversion using JavaScript's typed arrays for FP32

/*
function directFP16Conversion(number: number): number {
    const buffer = new ArrayBuffer(2);
    const float16 = new Float16Array(buffer);
    float16[0] = number;
    return float16[0];
}
*/
/*
function quantizeNumber(
    number: number, 
    config: QuantizationConfig
): number {
    // Handle special cases first
    if (isNaN(number)) return NaN;
    if (number === 0) return 0;
    if (number === Infinity) return Infinity;
    if (number === -Infinity) return -Infinity;
    
    // Get the IEEE 754 representation
    let parts = getNumberParts(number);
    
    console.log(`\nQuantizing ${number} to ${config.exponentBits}E${config.mantissaBits}M format`);
    console.log(`Original parts - Sign: ${parts.sign}, Exp: ${parts.exponent}, Mantissa: ${parts.mantissa}`);
    
    // Calculate the maximum and minimum representable exponents for target format
    const maxBiasedExp = (1 << config.exponentBits) - 2; // All 1s except infinity
    const minBiasedExp = 1; // Smallest normalized exponent
    const maxUnbiasedExp = maxBiasedExp - config.bias;
    const minUnbiasedExp = minBiasedExp - config.bias;
    
    console.log(`Target format exp range: ${minUnbiasedExp} to ${maxUnbiasedExp} (unbiased)`);
    
    // Check for overflow
    if (parts.exponent > maxUnbiasedExp) {
        console.log(`Overflow: exp ${parts.exponent} > max ${maxUnbiasedExp}`);
        return parts.sign ? -Infinity : Infinity;
    }
    
    // Check for underflow (becomes zero or denormal)
    if (parts.exponent < minUnbiasedExp) {
        console.log(`Underflow: exp ${parts.exponent} < min ${minUnbiasedExp}`);
        
        // Check if we can represent as denormal
        const denormalExp = minUnbiasedExp - 1;
        const denormalShift = denormalExp - parts.exponent;
        
        if (denormalShift > config.mantissaBits) {
            console.log(`Complete underflow to zero`);
            return parts.sign ? -0 : 0;
        }
        
        // Handle denormal numbers
        console.log(`Denormal representation needed, shift: ${denormalShift}`);
        const denormalMantissa = (1 + parts.mantissa) / Math.pow(2, denormalShift);
        const quantizedDenormalMantissa = Math.round(denormalMantissa * Math.pow(2, config.mantissaBits)) / Math.pow(2, config.mantissaBits);
        const result = (parts.sign ? -1 : 1) * quantizedDenormalMantissa * Math.pow(2, denormalExp);
        console.log(`Denormal result: ${result}`);
        return result;
    }
    
    // Normal case: quantize the mantissa
    // The mantissa represents the fractional part, we need to quantize it to the target precision
    const originalMantissaBits = 52; // Double precision
    const scaleFactor = Math.pow(2, config.mantissaBits);
    const originalScaleFactor = Math.pow(2, originalMantissaBits);
    
    // Convert mantissa to integer representation, quantize, then back to fractional
    const mantissaInt = Math.round(parts.mantissa * originalScaleFactor);
    const targetMantissaInt = Math.round(mantissaInt / Math.pow(2, originalMantissaBits - config.mantissaBits));
    const quantizedMantissa = targetMantissaInt / scaleFactor;
    
    console.log(`Mantissa quantization: ${parts.mantissa} -> ${quantizedMantissa}`);
    
    // Check for mantissa overflow (rounding up to 2.0)
    if (quantizedMantissa >= 1.0) {
        console.log(`Mantissa overflow, incrementing exponent`);
        const newExponent = parts.exponent + 1;
        
        // Check if this causes exponent overflow
        if (newExponent > maxUnbiasedExp) {
            console.log(`Exponent overflow after mantissa rounding`);
            return parts.sign ? -Infinity : Infinity;
        }
        
        // Mantissa becomes 0 (representing 1.0 in normalized form)
        const result = (parts.sign ? -1 : 1) * Math.pow(2, newExponent);
        console.log(`Result after mantissa overflow: ${result}`);
        return result;
    }
    
    // Reconstruct the number
    const result = (parts.sign ? -1 : 1) * (1 + quantizedMantissa) * Math.pow(2, parts.exponent);
    console.log(`Final result: ${result}`);
    
    return result;
}

// Example configurations
const FP64_CONFIG: QuantizationConfig = {
    exponentBits: 11,
    mantissaBits: 52,
    bias: 1023
};

const FP32_CONFIG: QuantizationConfig = {
    exponentBits: 8,
    mantissaBits: 23,
    bias: 127
};

const FP16_CONFIG: QuantizationConfig = {
    exponentBits: 5,
    mantissaBits: 10,
    bias: 15
};

const FP8_CONFIG: QuantizationConfig = {
    exponentBits: 4,
    mantissaBits: 3,
    bias: 7
};


// Test with different numbers
const testNumbers = [3.14159, 1.0, 0.1, 1000.0, -42.5, 1e-10, 1e10];

console.log('='.repeat(60));
console.log('FLOATING POINT QUANTIZATION TEST');
console.log('='.repeat(60));

testNumbers.forEach(number => {
    console.log(`\n${'='.repeat(40)}`);
    console.log(`Testing number: ${number}`);
    console.log(`${'='.repeat(40)}`);
    
    try {
        let fp32 = quantizeNumber(number, FP32_CONFIG);
        let fp16 = quantizeNumber(number, FP16_CONFIG);
        let fp8 = quantizeNumber(number, FP8_CONFIG);
        
        // Compare with direct FP32 conversion
        let directFP32 = directFP32Conversion(number);

        console.log(`\nResults:`);
        console.log(`Original: ${number}`);
        console.log(`FP32:     ${fp32}`);
        console.log(`FP32 (direct): ${directFP32}`);
        console.log(`FP16:     ${fp16}`);
        console.log(`FP8:      ${fp8}`);
        
        // Show differences
        console.log(`\nPrecision loss:`);
        console.log(`FP32 error: ${Math.abs(number - fp32).toExponential(3)}`);
        console.log(`FP32 direct error: ${Math.abs(number - directFP32).toExponential(3)}`);
        console.log(`FP16 error: ${Math.abs(number - fp16).toExponential(3)}`);
        console.log(`FP8 error:  ${Math.abs(number - fp8).toExponential(3)}`);
        
    } catch (error) {
        console.error(`Error processing ${number}:`, error);
    }
});

// Additional test for edge cases
console.log(`\n${'='.repeat(60)}`);
console.log('EDGE CASE TESTS');
console.log(`${'='.repeat(60)}`);

const edgeCases = [
    Number.MAX_VALUE,
    Number.MIN_VALUE,
    Number.EPSILON,
    1 + Number.EPSILON,
    Math.PI,
    Math.E,
    0.1 + 0.2, // Classic floating point issue
];

edgeCases.forEach(number => {
    console.log(`\nEdge case: ${number}`);
    try {
        let fp32 = quantizeNumber(number, FP32_CONFIG);
        let directFP32 = directFP32Conversion(number);
        console.log(`Custom FP32: ${fp32}`);
        console.log(`Direct FP32: ${directFP32}`);
        console.log(`Match: ${fp32 === directFP32}`);
    } catch (error) {
        console.error(`Error:`, error);
    }
});
*/
