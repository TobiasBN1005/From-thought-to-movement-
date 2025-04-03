################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
S_SRCS += \
../EEG_code/Core/Startup/startup_stm32f303k8tx.s 

OBJS += \
./EEG_code/Core/Startup/startup_stm32f303k8tx.o 

S_DEPS += \
./EEG_code/Core/Startup/startup_stm32f303k8tx.d 


# Each subdirectory must supply rules for building sources it contributes
EEG_code/Core/Startup/%.o: ../EEG_code/Core/Startup/%.s EEG_code/Core/Startup/subdir.mk
	arm-none-eabi-gcc -mcpu=cortex-m4 -g3 -DDEBUG -c -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@" "$<"

clean: clean-EEG_code-2f-Core-2f-Startup

clean-EEG_code-2f-Core-2f-Startup:
	-$(RM) ./EEG_code/Core/Startup/startup_stm32f303k8tx.d ./EEG_code/Core/Startup/startup_stm32f303k8tx.o

.PHONY: clean-EEG_code-2f-Core-2f-Startup

