	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 11
	.section	__TEXT,__literal8,8byte_literals
	.align	3
LCPI0_0:
	.quad	4613937818241073152     ## double 3
LCPI0_3:
	.quad	4696837146684686336     ## double 1.0E+6
	.section	__TEXT,__literal16,16byte_literals
	.align	4
LCPI0_1:
	.long	1127219200              ## 0x43300000
	.long	1160773632              ## 0x45300000
	.long	0                       ## 0x0
	.long	0                       ## 0x0
LCPI0_2:
	.quad	4841369599423283200     ## double 4.503600e+15
	.quad	4985484787499139072     ## double 1.934281e+25
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.align	4, 0x90
_main:                                  ## @main
Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception0
## BB#0:
	pushq	%rbp
Ltmp16:
	.cfi_def_cfa_offset 16
Ltmp17:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp18:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$88, %rsp
Ltmp19:
	.cfi_offset %rbx, -56
Ltmp20:
	.cfi_offset %r12, -48
Ltmp21:
	.cfi_offset %r13, -40
Ltmp22:
	.cfi_offset %r14, -32
Ltmp23:
	.cfi_offset %r15, -24
	movq	8(%rsi), %rdi
	callq	_atoi
	movl	%eax, %r15d
	movslq	%r15d, %rax
	movl	$8, %ecx
	mulq	%rcx
	movq	$-1, %rbx
	cmovnoq	%rax, %rbx
	movq	%rbx, %rdi
	callq	__Znam
	movq	%rax, -88(%rbp)         ## 8-byte Spill
	movq	%rbx, %rdi
	callq	__Znam
	movq	%rax, %r12
	movq	%rbx, %rdi
	callq	__Znam
	movq	%rax, %r13
	callq	_clock
	movq	%rax, %rbx
	xorpd	%xmm2, %xmm2
	testl	%r15d, %r15d
	jle	LBB0_20
## BB#1:                                ## %.lr.ph19
	movq	%rbx, -112(%rbp)        ## 8-byte Spill
	leal	-1(%r15), %r14d
	leaq	8(,%r14,8), %rdx
	leaq	_.memset_pattern(%rip), %rsi
	movq	%r13, %rdi
	movq	%r13, -96(%rbp)         ## 8-byte Spill
	callq	_memset_pattern16
	xorl	%r13d, %r13d
	testb	$1, %r15b
	movq	%r15, -104(%rbp)        ## 8-byte Spill
	je	LBB0_3
## BB#2:
	xorpd	%xmm0, %xmm0
	callq	_cos
	movq	-88(%rbp), %rax         ## 8-byte Reload
	movsd	%xmm0, (%rax)
	movsd	LCPI0_0(%rip), %xmm0    ## xmm0 = mem[0],zero
	callq	_sin
	movsd	%xmm0, (%r12)
	movl	$1, %r13d
LBB0_3:                                 ## %.lr.ph19.split
	testl	%r14d, %r14d
	movq	%r12, %r15
	movq	-88(%rbp), %r12         ## 8-byte Reload
	je	LBB0_4
	.align	4, 0x90
LBB0_30:                                ## =>This Inner Loop Header: Depth=1
	xorps	%xmm0, %xmm0
	cvtsi2sdl	%r13d, %xmm0
	movsd	%xmm0, -80(%rbp)        ## 8-byte Spill
	callq	_cos
	movsd	%xmm0, (%r12,%r13,8)
	movsd	LCPI0_0(%rip), %xmm0    ## xmm0 = mem[0],zero
	movsd	-80(%rbp), %xmm1        ## 8-byte Reload
                                        ## xmm1 = mem[0],zero
	addsd	%xmm0, %xmm1
	movapd	%xmm1, %xmm0
	callq	_sin
	movsd	%xmm0, (%r15,%r13,8)
	leal	1(%r13), %ebx
	xorps	%xmm0, %xmm0
	cvtsi2sdl	%ebx, %xmm0
	movsd	%xmm0, -80(%rbp)        ## 8-byte Spill
	callq	_cos
	movsd	%xmm0, 8(%r12,%r13,8)
	movsd	-80(%rbp), %xmm0        ## 8-byte Reload
                                        ## xmm0 = mem[0],zero
	addsd	LCPI0_0(%rip), %xmm0
	callq	_sin
	movsd	%xmm0, 8(%r15,%r13,8)
	addq	$2, %r13
	cmpl	%r14d, %ebx
	jne	LBB0_30
LBB0_4:                                 ## %.preheader12
	movq	-104(%rbp), %rax        ## 8-byte Reload
	cmpl	$2, %eax
	jl	LBB0_5
## BB#6:                                ## %.lr.ph16
	movq	-96(%rbp), %r13         ## 8-byte Reload
	movsd	(%r13), %xmm0           ## xmm0 = mem[0],zero
	leal	-2(%rax), %r8d
	movq	%rax, %rdx
	movl	$1, %ecx
	testb	$1, %r14b
	movq	-88(%rbp), %rax         ## 8-byte Reload
	movq	%r15, %r12
	xorpd	%xmm2, %xmm2
	je	LBB0_8
## BB#7:
	movsd	8(%rax), %xmm1          ## xmm1 = mem[0],zero
	addsd	(%r12), %xmm1
	addsd	%xmm1, %xmm0
	movsd	%xmm0, 8(%r13)
	movl	$2, %ecx
LBB0_8:                                 ## %.lr.ph16.split
	testl	%r8d, %r8d
	movq	%rdx, %r9
	je	LBB0_11
## BB#9:                                ## %.lr.ph16.split.split
	leaq	(%r12,%rcx,8), %rdx
	leaq	8(%rax,%rcx,8), %rsi
	leaq	8(%r13,%rcx,8), %rdi
	leal	1(%r9), %eax
	leal	1(%rcx), %ebx
	subl	%ebx, %eax
	xorl	%ebx, %ebx
	.align	4, 0x90
LBB0_10:                                ## =>This Inner Loop Header: Depth=1
	movsd	-8(%rsi,%rbx,8), %xmm1  ## xmm1 = mem[0],zero
	addsd	-8(%rdx,%rbx,8), %xmm1
	addsd	%xmm0, %xmm1
	movsd	%xmm1, -8(%rdi,%rbx,8)
	movsd	(%rsi,%rbx,8), %xmm0    ## xmm0 = mem[0],zero
	addsd	(%rdx,%rbx,8), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, (%rdi,%rbx,8)
	addq	$2, %rcx
	addq	$2, %rbx
	cmpl	%ebx, %eax
	jne	LBB0_10
LBB0_11:                                ## %.preheader
	cmpl	$2, %r9d
	movq	-112(%rbp), %rbx        ## 8-byte Reload
	jl	LBB0_20
## BB#12:                               ## %.lr.ph
	movl	$1, %ecx
	testb	$3, %r14b
	je	LBB0_13
## BB#14:                               ## %.preheader32
	andq	$3, %r14
	xorpd	%xmm2, %xmm2
	xorl	%ecx, %ecx
	.align	4, 0x90
LBB0_15:                                ## =>This Inner Loop Header: Depth=1
	movapd	%xmm2, %xmm0
	movsd	8(%r13,%rcx,8), %xmm2   ## xmm2 = mem[0],zero
	mulsd	%xmm2, %xmm2
	addsd	%xmm0, %xmm2
	incq	%rcx
	cmpl	%ecx, %r14d
	jne	LBB0_15
## BB#16:                               ## %.lr.ph.split.loopexit
	incq	%rcx
	jmp	LBB0_17
LBB0_5:
	movq	%r15, %r12
	movq	-96(%rbp), %r13         ## 8-byte Reload
	xorpd	%xmm2, %xmm2
	movq	-112(%rbp), %rbx        ## 8-byte Reload
	jmp	LBB0_20
LBB0_13:
	xorpd	%xmm2, %xmm2
LBB0_17:                                ## %.lr.ph.split
	cmpl	$3, %r8d
	jb	LBB0_20
## BB#18:                               ## %.lr.ph.split.split
	leaq	24(%r13,%rcx,8), %rax
	addl	$3, %r9d
	leal	3(%rcx), %edx
	subl	%edx, %r9d
	.align	4, 0x90
LBB0_19:                                ## =>This Inner Loop Header: Depth=1
	movsd	-24(%rax), %xmm0        ## xmm0 = mem[0],zero
	movsd	-16(%rax), %xmm1        ## xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm0
	addsd	%xmm2, %xmm0
	mulsd	%xmm1, %xmm1
	addsd	%xmm0, %xmm1
	movsd	-8(%rax), %xmm0         ## xmm0 = mem[0],zero
	mulsd	%xmm0, %xmm0
	addsd	%xmm1, %xmm0
	movsd	(%rax), %xmm2           ## xmm2 = mem[0],zero
	mulsd	%xmm2, %xmm2
	addsd	%xmm0, %xmm2
	addq	$4, %rcx
	addq	$32, %rax
	addl	$-4, %r9d
	jne	LBB0_19
LBB0_20:                                ## %._crit_edge
	movsd	%xmm2, -96(%rbp)        ## 8-byte Spill
	movq	%r13, %r14
	callq	_clock
	subq	%rbx, %rax
	movd	%rax, %xmm0
	punpckldq	LCPI0_1(%rip), %xmm0 ## xmm0 = xmm0[0],mem[0],xmm0[1],mem[1]
	subpd	LCPI0_2(%rip), %xmm0
	haddpd	%xmm0, %xmm0
	divsd	LCPI0_3(%rip), %xmm0
	movaps	%xmm0, -80(%rbp)        ## 16-byte Spill
	movq	__ZNSt3__14coutE@GOTPCREL(%rip), %r13
	movq	(%r13), %rax
	movq	-24(%rax), %rax
	orl	$17408, 8(%rax,%r13)    ## imm = 0x4400
	movq	(%r13), %rax
	movq	-24(%rax), %rax
	movq	$10, 16(%rax,%r13)
	movq	(%r13), %rax
	movq	-24(%rax), %rax
	movq	$20, 24(%rax,%r13)
	leaq	L_.str(%rip), %rsi
	movl	$31, %edx
	movq	%r13, %rdi
	callq	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	movq	%rax, %rdi
	movapd	-80(%rbp), %xmm0        ## 16-byte Reload
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsEd
	movq	%rax, %rbx
	movq	(%rbx), %rax
	movq	-24(%rax), %rsi
	addq	%rbx, %rsi
	leaq	-48(%rbp), %r15
	movq	%r15, %rdi
	callq	__ZNKSt3__18ios_base6getlocEv
Ltmp0:
	movq	__ZNSt3__15ctypeIcE2idE@GOTPCREL(%rip), %rsi
	movq	%r15, %rdi
	callq	__ZNKSt3__16locale9use_facetERNS0_2idE
Ltmp1:
## BB#21:
	movq	(%rax), %rcx
	movq	56(%rcx), %rcx
Ltmp2:
	movl	$10, %esi
	movq	%rax, %rdi
	callq	*%rcx
	movb	%al, %r15b
Ltmp3:
## BB#22:                               ## %_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5widenEc.exit6
	leaq	-48(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
	movsbl	%r15b, %esi
	movq	%rbx, %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE3putEc
	movq	%rbx, %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE5flushEv
	movq	(%r13), %rax
	movq	-24(%rax), %rax
	movq	$10, 16(%rax,%r13)
	movq	(%r13), %rax
	movq	-24(%rax), %rax
	movq	$20, 24(%rax,%r13)
	leaq	L_.str1(%rip), %rsi
	movl	$10, %edx
	movq	%r13, %rdi
	callq	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	movq	%rax, %rdi
	movsd	-96(%rbp), %xmm0        ## 8-byte Reload
                                        ## xmm0 = mem[0],zero
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsEd
	movq	%rax, %rbx
	movq	(%rbx), %rax
	movq	-24(%rax), %rsi
	addq	%rbx, %rsi
	leaq	-56(%rbp), %r15
	movq	%r15, %rdi
	callq	__ZNKSt3__18ios_base6getlocEv
Ltmp8:
	movq	__ZNSt3__15ctypeIcE2idE@GOTPCREL(%rip), %rsi
	movq	%r15, %rdi
	callq	__ZNKSt3__16locale9use_facetERNS0_2idE
Ltmp9:
## BB#23:
	movq	(%rax), %rcx
	movq	56(%rcx), %rcx
Ltmp10:
	movl	$10, %esi
	movq	%rax, %rdi
	callq	*%rcx
	movb	%al, %r15b
Ltmp11:
## BB#24:                               ## %_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5widenEc.exit
	leaq	-56(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
	movsbl	%r15b, %esi
	movq	%rbx, %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE3putEc
	movq	%rbx, %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE5flushEv
	movq	-88(%rbp), %rdi         ## 8-byte Reload
	callq	__ZdaPv
	movq	%r12, %rdi
	callq	__ZdaPv
	movq	%r14, %rdi
	callq	__ZdaPv
	xorl	%eax, %eax
	addq	$88, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
LBB0_25:
Ltmp4:
	movq	%rax, %rbx
Ltmp5:
	leaq	-48(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
Ltmp6:
	jmp	LBB0_26
LBB0_27:
Ltmp7:
	movq	%rax, %rdi
	callq	___clang_call_terminate
LBB0_28:
Ltmp12:
	movq	%rax, %rbx
Ltmp13:
	leaq	-56(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
Ltmp14:
LBB0_26:                                ## %unwind_resume
	movq	%rbx, %rdi
	callq	__Unwind_Resume
LBB0_29:
Ltmp15:
	movq	%rax, %rdi
	callq	___clang_call_terminate
Lfunc_end0:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.align	2
GCC_except_table0:
Lexception0:
	.byte	255                     ## @LPStart Encoding = omit
	.byte	155                     ## @TType Encoding = indirect pcrel sdata4
	.asciz	"\360"                  ## @TType base offset
	.byte	3                       ## Call site Encoding = udata4
	.byte	104                     ## Call site table length
Lset0 = Lfunc_begin0-Lfunc_begin0       ## >> Call Site 1 <<
	.long	Lset0
Lset1 = Ltmp0-Lfunc_begin0              ##   Call between Lfunc_begin0 and Ltmp0
	.long	Lset1
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset2 = Ltmp0-Lfunc_begin0              ## >> Call Site 2 <<
	.long	Lset2
Lset3 = Ltmp3-Ltmp0                     ##   Call between Ltmp0 and Ltmp3
	.long	Lset3
Lset4 = Ltmp4-Lfunc_begin0              ##     jumps to Ltmp4
	.long	Lset4
	.byte	0                       ##   On action: cleanup
Lset5 = Ltmp3-Lfunc_begin0              ## >> Call Site 3 <<
	.long	Lset5
Lset6 = Ltmp8-Ltmp3                     ##   Call between Ltmp3 and Ltmp8
	.long	Lset6
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset7 = Ltmp8-Lfunc_begin0              ## >> Call Site 4 <<
	.long	Lset7
Lset8 = Ltmp11-Ltmp8                    ##   Call between Ltmp8 and Ltmp11
	.long	Lset8
Lset9 = Ltmp12-Lfunc_begin0             ##     jumps to Ltmp12
	.long	Lset9
	.byte	0                       ##   On action: cleanup
Lset10 = Ltmp11-Lfunc_begin0            ## >> Call Site 5 <<
	.long	Lset10
Lset11 = Ltmp5-Ltmp11                   ##   Call between Ltmp11 and Ltmp5
	.long	Lset11
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset12 = Ltmp5-Lfunc_begin0             ## >> Call Site 6 <<
	.long	Lset12
Lset13 = Ltmp6-Ltmp5                    ##   Call between Ltmp5 and Ltmp6
	.long	Lset13
Lset14 = Ltmp7-Lfunc_begin0             ##     jumps to Ltmp7
	.long	Lset14
	.byte	1                       ##   On action: 1
Lset15 = Ltmp13-Lfunc_begin0            ## >> Call Site 7 <<
	.long	Lset15
Lset16 = Ltmp14-Ltmp13                  ##   Call between Ltmp13 and Ltmp14
	.long	Lset16
Lset17 = Ltmp15-Lfunc_begin0            ##     jumps to Ltmp15
	.long	Lset17
	.byte	1                       ##   On action: 1
Lset18 = Ltmp14-Lfunc_begin0            ## >> Call Site 8 <<
	.long	Lset18
Lset19 = Lfunc_end0-Ltmp14              ##   Call between Ltmp14 and Lfunc_end0
	.long	Lset19
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
	.byte	1                       ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
	.byte	0                       ##   No further actions
                                        ## >> Catch TypeInfos <<
	.long	0                       ## TypeInfo 1
	.align	2

	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.globl	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	.weak_def_can_be_hidden	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	.align	4, 0x90
__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m: ## @_ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception1
## BB#0:
	pushq	%rbp
Ltmp54:
	.cfi_def_cfa_offset 16
Ltmp55:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp56:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$56, %rsp
Ltmp57:
	.cfi_offset %rbx, -56
Ltmp58:
	.cfi_offset %r12, -48
Ltmp59:
	.cfi_offset %r13, -40
Ltmp60:
	.cfi_offset %r14, -32
Ltmp61:
	.cfi_offset %r15, -24
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %rbx
Ltmp24:
	leaq	-64(%rbp), %rdi
	movq	%rbx, %rsi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryC1ERS3_
Ltmp25:
## BB#1:
	cmpb	$0, -64(%rbp)
	je	LBB1_13
## BB#2:
	movq	(%rbx), %rax
	movq	-24(%rax), %r12
	movq	40(%r12,%rbx), %rdi
	movl	$176, %eax
	andl	8(%r12,%rbx), %eax
	cmpl	$32, %eax
	movq	%r15, %r13
	jne	LBB1_4
## BB#3:
	leaq	(%r15,%r14), %r13
LBB1_4:
	leaq	(%rbx,%r12), %r8
	movl	144(%rbx,%r12), %eax
	cmpl	$-1, %eax
	jne	LBB1_10
## BB#5:
Ltmp26:
	movq	%rdi, -72(%rbp)         ## 8-byte Spill
	leaq	-48(%rbp), %rdi
	movq	%r8, %rsi
	movq	%r8, -80(%rbp)          ## 8-byte Spill
	callq	__ZNKSt3__18ios_base6getlocEv
Ltmp27:
## BB#6:                                ## %.noexc
Ltmp28:
	movq	__ZNSt3__15ctypeIcE2idE@GOTPCREL(%rip), %rsi
	leaq	-48(%rbp), %rdi
	callq	__ZNKSt3__16locale9use_facetERNS0_2idE
Ltmp29:
## BB#7:
	movq	(%rax), %rcx
	movq	56(%rcx), %rcx
Ltmp30:
	movl	$32, %esi
	movq	%rax, %rdi
	callq	*%rcx
	movb	%al, -81(%rbp)          ## 1-byte Spill
Ltmp31:
## BB#8:                                ## %_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5widenEc.exit.i
Ltmp36:
	leaq	-48(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
Ltmp37:
## BB#9:                                ## %.noexc1
	movsbl	-81(%rbp), %eax         ## 1-byte Folded Reload
	movl	%eax, 144(%rbx,%r12)
	movq	-72(%rbp), %rdi         ## 8-byte Reload
	movq	-80(%rbp), %r8          ## 8-byte Reload
LBB1_10:
	addq	%r15, %r14
Ltmp38:
	movsbl	%al, %r9d
	movq	%r15, %rsi
	movq	%r13, %rdx
	movq	%r14, %rcx
	callq	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
Ltmp39:
## BB#11:
	testq	%rax, %rax
	jne	LBB1_13
## BB#12:
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	leaq	(%rbx,%rax), %rdi
	movl	32(%rbx,%rax), %esi
	orl	$5, %esi
Ltmp40:
	callq	__ZNSt3__18ios_base5clearEj
Ltmp41:
LBB1_13:                                ## %_ZNSt3__19basic_iosIcNS_11char_traitsIcEEE8setstateEj.exit
Ltmp45:
	leaq	-64(%rbp), %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev
Ltmp46:
LBB1_21:
	movq	%rbx, %rax
	addq	$56, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
LBB1_18:
Ltmp47:
	movq	%rax, %r14
	jmp	LBB1_19
LBB1_16:
Ltmp42:
	movq	%rax, %r14
	jmp	LBB1_17
LBB1_14:
Ltmp32:
	movq	%rax, %r14
Ltmp33:
	leaq	-48(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
Ltmp34:
LBB1_17:                                ## %.body
Ltmp43:
	leaq	-64(%rbp), %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev
Ltmp44:
LBB1_19:
	movq	%r14, %rdi
	callq	___cxa_begin_catch
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	addq	-24(%rax), %rdi
Ltmp48:
	callq	__ZNSt3__18ios_base33__set_badbit_and_consider_rethrowEv
Ltmp49:
## BB#20:
	callq	___cxa_end_catch
	jmp	LBB1_21
LBB1_22:
Ltmp50:
	movq	%rax, %rbx
Ltmp51:
	callq	___cxa_end_catch
Ltmp52:
## BB#23:
	movq	%rbx, %rdi
	callq	__Unwind_Resume
LBB1_24:
Ltmp53:
	movq	%rax, %rdi
	callq	___clang_call_terminate
LBB1_15:
Ltmp35:
	movq	%rax, %rdi
	callq	___clang_call_terminate
Lfunc_end1:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.align	2
GCC_except_table1:
Lexception1:
	.byte	255                     ## @LPStart Encoding = omit
	.byte	155                     ## @TType Encoding = indirect pcrel sdata4
	.asciz	"\245\201\200\200"      ## @TType base offset
	.byte	3                       ## Call site Encoding = udata4
	.ascii	"\234\001"              ## Call site table length
Lset20 = Ltmp24-Lfunc_begin1            ## >> Call Site 1 <<
	.long	Lset20
Lset21 = Ltmp25-Ltmp24                  ##   Call between Ltmp24 and Ltmp25
	.long	Lset21
Lset22 = Ltmp47-Lfunc_begin1            ##     jumps to Ltmp47
	.long	Lset22
	.byte	1                       ##   On action: 1
Lset23 = Ltmp26-Lfunc_begin1            ## >> Call Site 2 <<
	.long	Lset23
Lset24 = Ltmp27-Ltmp26                  ##   Call between Ltmp26 and Ltmp27
	.long	Lset24
Lset25 = Ltmp42-Lfunc_begin1            ##     jumps to Ltmp42
	.long	Lset25
	.byte	1                       ##   On action: 1
Lset26 = Ltmp28-Lfunc_begin1            ## >> Call Site 3 <<
	.long	Lset26
Lset27 = Ltmp31-Ltmp28                  ##   Call between Ltmp28 and Ltmp31
	.long	Lset27
Lset28 = Ltmp32-Lfunc_begin1            ##     jumps to Ltmp32
	.long	Lset28
	.byte	1                       ##   On action: 1
Lset29 = Ltmp36-Lfunc_begin1            ## >> Call Site 4 <<
	.long	Lset29
Lset30 = Ltmp41-Ltmp36                  ##   Call between Ltmp36 and Ltmp41
	.long	Lset30
Lset31 = Ltmp42-Lfunc_begin1            ##     jumps to Ltmp42
	.long	Lset31
	.byte	1                       ##   On action: 1
Lset32 = Ltmp45-Lfunc_begin1            ## >> Call Site 5 <<
	.long	Lset32
Lset33 = Ltmp46-Ltmp45                  ##   Call between Ltmp45 and Ltmp46
	.long	Lset33
Lset34 = Ltmp47-Lfunc_begin1            ##     jumps to Ltmp47
	.long	Lset34
	.byte	1                       ##   On action: 1
Lset35 = Ltmp33-Lfunc_begin1            ## >> Call Site 6 <<
	.long	Lset35
Lset36 = Ltmp34-Ltmp33                  ##   Call between Ltmp33 and Ltmp34
	.long	Lset36
Lset37 = Ltmp35-Lfunc_begin1            ##     jumps to Ltmp35
	.long	Lset37
	.byte	1                       ##   On action: 1
Lset38 = Ltmp43-Lfunc_begin1            ## >> Call Site 7 <<
	.long	Lset38
Lset39 = Ltmp44-Ltmp43                  ##   Call between Ltmp43 and Ltmp44
	.long	Lset39
Lset40 = Ltmp53-Lfunc_begin1            ##     jumps to Ltmp53
	.long	Lset40
	.byte	1                       ##   On action: 1
Lset41 = Ltmp44-Lfunc_begin1            ## >> Call Site 8 <<
	.long	Lset41
Lset42 = Ltmp48-Ltmp44                  ##   Call between Ltmp44 and Ltmp48
	.long	Lset42
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset43 = Ltmp48-Lfunc_begin1            ## >> Call Site 9 <<
	.long	Lset43
Lset44 = Ltmp49-Ltmp48                  ##   Call between Ltmp48 and Ltmp49
	.long	Lset44
Lset45 = Ltmp50-Lfunc_begin1            ##     jumps to Ltmp50
	.long	Lset45
	.byte	0                       ##   On action: cleanup
Lset46 = Ltmp49-Lfunc_begin1            ## >> Call Site 10 <<
	.long	Lset46
Lset47 = Ltmp51-Ltmp49                  ##   Call between Ltmp49 and Ltmp51
	.long	Lset47
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset48 = Ltmp51-Lfunc_begin1            ## >> Call Site 11 <<
	.long	Lset48
Lset49 = Ltmp52-Ltmp51                  ##   Call between Ltmp51 and Ltmp52
	.long	Lset49
Lset50 = Ltmp53-Lfunc_begin1            ##     jumps to Ltmp53
	.long	Lset50
	.byte	1                       ##   On action: 1
Lset51 = Ltmp52-Lfunc_begin1            ## >> Call Site 12 <<
	.long	Lset51
Lset52 = Lfunc_end1-Ltmp52              ##   Call between Ltmp52 and Lfunc_end1
	.long	Lset52
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
	.byte	1                       ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
	.byte	0                       ##   No further actions
                                        ## >> Catch TypeInfos <<
	.long	0                       ## TypeInfo 1
	.align	2

	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.private_extern	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
	.globl	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
	.weak_def_can_be_hidden	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
	.align	4, 0x90
__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_: ## @_ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception2
## BB#0:
	pushq	%rbp
Ltmp68:
	.cfi_def_cfa_offset 16
Ltmp69:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp70:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$40, %rsp
Ltmp71:
	.cfi_offset %rbx, -56
Ltmp72:
	.cfi_offset %r12, -48
Ltmp73:
	.cfi_offset %r13, -40
Ltmp74:
	.cfi_offset %r14, -32
Ltmp75:
	.cfi_offset %r15, -24
	movq	%r8, %r14
	movq	%rcx, %r13
	movq	%rdi, %rbx
	xorl	%eax, %eax
	testq	%rbx, %rbx
	je	LBB2_12
## BB#1:
	movq	%r13, %rax
	subq	%rsi, %rax
	movq	24(%r14), %rcx
	xorl	%r15d, %r15d
	subq	%rax, %rcx
	cmovgq	%rcx, %r15
	movq	%rdx, %r12
	subq	%rsi, %r12
	testq	%r12, %r12
	jle	LBB2_3
## BB#2:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	movq	%r13, -80(%rbp)         ## 8-byte Spill
	movq	%rdx, -72(%rbp)         ## 8-byte Spill
	movq	%r12, %rdx
	movl	%r9d, %r13d
	callq	*96(%rax)
	movl	%r13d, %r9d
	movq	-72(%rbp), %rdx         ## 8-byte Reload
	movq	-80(%rbp), %r13         ## 8-byte Reload
	movq	%rax, %rcx
	xorl	%eax, %eax
	cmpq	%r12, %rcx
	jne	LBB2_12
LBB2_3:
	testq	%r15, %r15
	jle	LBB2_9
## BB#4:
	movq	%rdx, -72(%rbp)         ## 8-byte Spill
	movq	%r14, %r12
	movsbl	%r9b, %edx
	leaq	-64(%rbp), %rdi
	movq	%r15, %rsi
	callq	__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEmc
	testb	$1, -64(%rbp)
	jne	LBB2_5
## BB#6:
	leaq	-63(%rbp), %rsi
	jmp	LBB2_7
LBB2_5:
	movq	-48(%rbp), %rsi
LBB2_7:                                 ## %_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit
	movq	(%rbx), %rax
	movq	96(%rax), %rax
Ltmp62:
	movq	%rbx, %rdi
	movq	%r15, %rdx
	callq	*%rax
	movq	%rax, %r14
Ltmp63:
## BB#8:                                ## %_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKcl.exit
	leaq	-64(%rbp), %rdi
	callq	__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev
	xorl	%eax, %eax
	cmpq	%r15, %r14
	movq	%r12, %r14
	movq	-72(%rbp), %rdx         ## 8-byte Reload
	jne	LBB2_12
LBB2_9:                                 ## %.thread
	subq	%rdx, %r13
	testq	%r13, %r13
	jle	LBB2_11
## BB#10:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	movq	%rdx, %rsi
	movq	%r13, %rdx
	callq	*96(%rax)
	movq	%rax, %rcx
	xorl	%eax, %eax
	cmpq	%r13, %rcx
	jne	LBB2_12
LBB2_11:
	movq	$0, 24(%r14)
	movq	%rbx, %rax
LBB2_12:
	addq	$40, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
LBB2_13:
Ltmp64:
	movq	%rax, %rbx
Ltmp65:
	leaq	-64(%rbp), %rdi
	callq	__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev
Ltmp66:
## BB#14:
	movq	%rbx, %rdi
	callq	__Unwind_Resume
LBB2_15:
Ltmp67:
	movq	%rax, %rdi
	callq	___clang_call_terminate
Lfunc_end2:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.align	2
GCC_except_table2:
Lexception2:
	.byte	255                     ## @LPStart Encoding = omit
	.byte	155                     ## @TType Encoding = indirect pcrel sdata4
	.byte	73                      ## @TType base offset
	.byte	3                       ## Call site Encoding = udata4
	.byte	65                      ## Call site table length
Lset53 = Lfunc_begin2-Lfunc_begin2      ## >> Call Site 1 <<
	.long	Lset53
Lset54 = Ltmp62-Lfunc_begin2            ##   Call between Lfunc_begin2 and Ltmp62
	.long	Lset54
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset55 = Ltmp62-Lfunc_begin2            ## >> Call Site 2 <<
	.long	Lset55
Lset56 = Ltmp63-Ltmp62                  ##   Call between Ltmp62 and Ltmp63
	.long	Lset56
Lset57 = Ltmp64-Lfunc_begin2            ##     jumps to Ltmp64
	.long	Lset57
	.byte	0                       ##   On action: cleanup
Lset58 = Ltmp63-Lfunc_begin2            ## >> Call Site 3 <<
	.long	Lset58
Lset59 = Ltmp65-Ltmp63                  ##   Call between Ltmp63 and Ltmp65
	.long	Lset59
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset60 = Ltmp65-Lfunc_begin2            ## >> Call Site 4 <<
	.long	Lset60
Lset61 = Ltmp66-Ltmp65                  ##   Call between Ltmp65 and Ltmp66
	.long	Lset61
Lset62 = Ltmp67-Lfunc_begin2            ##     jumps to Ltmp67
	.long	Lset62
	.byte	1                       ##   On action: 1
Lset63 = Ltmp66-Lfunc_begin2            ## >> Call Site 5 <<
	.long	Lset63
Lset64 = Lfunc_end2-Ltmp66              ##   Call between Ltmp66 and Lfunc_end2
	.long	Lset64
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
	.byte	1                       ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
	.byte	0                       ##   No further actions
                                        ## >> Catch TypeInfos <<
	.long	0                       ## TypeInfo 1
	.align	2

	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.private_extern	___clang_call_terminate
	.globl	___clang_call_terminate
	.weak_def_can_be_hidden	___clang_call_terminate
	.align	4, 0x90
___clang_call_terminate:                ## @__clang_call_terminate
## BB#0:
	pushq	%rbp
	movq	%rsp, %rbp
	callq	___cxa_begin_catch
	callq	__ZSt9terminatev

	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"Time used  for vector addition="

L_.str1:                                ## @.str1
	.asciz	"Final sum="

	.section	__TEXT,__const
	.align	4                       ## @.memset_pattern
_.memset_pattern:
	.quad	4607182418800017408     ## double 1.000000e+00
	.quad	4607182418800017408     ## double 1.000000e+00


.subsections_via_symbols
