################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../EEG_code/Core/Src/main.c \
../EEG_code/Core/Src/stm32f3xx_hal_msp.c \
../EEG_code/Core/Src/stm32f3xx_it.c \
../EEG_code/Core/Src/syscalls.c \
../EEG_code/Core/Src/sysmem.c \
../EEG_code/Core/Src/system_stm32f3xx.c 

OBJS += \
./EEG_code/Core/Src/main.o \
./EEG_code/Core/Src/stm32f3xx_hal_msp.o \
./EEG_code/Core/Src/stm32f3xx_it.o \
./EEG_code/Core/Src/syscalls.o \
./EEG_code/Core/Src/sysmem.o \
./EEG_code/Core/Src/system_stm32f3xx.o 

C_DEPS += \
./EEG_code/Core/Src/main.d \
./EEG_code/Core/Src/stm32f3xx_hal_msp.d \
./EEG_code/Core/Src/stm32f3xx_it.d \
./EEG_code/Core/Src/syscalls.d \
./EEG_code/Core/Src/sysmem.d \
./EEG_code/Core/Src/system_stm32f3xx.d 


# Each subdirectory must supply rules for building sources it contributes
EEG_code/Core/Src/%.o EEG_code/Core/Src/%.su EEG_code/Core/Src/%.cyclo: ../EEG_code/Core/Src/%.c EEG_code/Core/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F303x8 -c -I../Core/Inc -I../Drivers/STM32F3xx_HAL_Driver/Inc -I../Drivers/STM32F3xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F3xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-EEG_code-2f-Core-2f-Src

clean-EEG_code-2f-Core-2f-Src:
	-$(RM) ./EEG_code/Core/Src/main.cyclo ./EEG_code/Core/Src/main.d ./EEG_code/Core/Src/main.o ./EEG_code/Core/Src/main.su ./EEG_code/Core/Src/stm32f3xx_hal_msp.cyclo ./EEG_code/Core/Src/stm32f3xx_hal_msp.d ./EEG_code/Core/Src/stm32f3xx_hal_msp.o ./EEG_code/Core/Src/stm32f3xx_hal_msp.su ./EEG_code/Core/Src/stm32f3xx_it.cyclo ./EEG_code/Core/Src/stm32f3xx_it.d ./EEG_code/Core/Src/stm32f3xx_it.o ./EEG_code/Core/Src/stm32f3xx_it.su ./EEG_code/Core/Src/syscalls.cyclo ./EEG_code/Core/Src/syscalls.d ./EEG_code/Core/Src/syscalls.o ./EEG_code/Core/Src/syscalls.su ./EEG_code/Core/Src/sysmem.cyclo ./EEG_code/Core/Src/sysmem.d ./EEG_code/Core/Src/sysmem.o ./EEG_code/Core/Src/sysmem.su ./EEG_code/Core/Src/system_stm32f3xx.cyclo ./EEG_code/Core/Src/system_stm32f3xx.d ./EEG_code/Core/Src/system_stm32f3xx.o ./EEG_code/Core/Src/system_stm32f3xx.su

.PHONY: clean-EEG_code-2f-Core-2f-Src

